from typing import Annotated, Optional
import json
import os
from datetime import datetime
from enum import Enum

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship, delete
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from contextlib import asynccontextmanager
import xxhash

from map import _create_and_save_image

CACHE_DIR = '.cache'


class SplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class DatasetSequenceLink(SQLModel, table=True):
    sequence_id: int | None = Field(default=None, foreign_key='sequence.id', primary_key=True)
    dataset_id: int | None = Field(default=None, foreign_key='dataset.id', primary_key=True)
    split_type: SplitType = Field(default=SplitType.TRAIN)


class SequenceBase(SQLModel):
    name: str = Field(index=True, unique=True)
    duration: float
    gps: str | None = Field(default=None)
    labeled_frames: str = "[]"

class Sequence(SequenceBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    uncertainty_measurements: list['SequenceUncertainty'] = Relationship(
        back_populates='sequence',
        sa_relationship_kwargs={
            'foreign_keys': '[SequenceUncertainty.sequence_id]',
            'order_by': 'desc(SequenceUncertainty.created_at)',
            'cascade': 'all, delete'
        }
    )
    datasets: list['Dataset'] = Relationship(back_populates='sequences', link_model=DatasetSequenceLink)

    @property
    def latest_uncertainty(self) -> Optional['SequenceUncertainty']:
        if self.uncertainty_measurements:
            return self.uncertainty_measurements[0]
        return None

class SequencePublic(SequenceBase):
    id: int
    latest_uncertainty: Optional['SequenceUncertaintyPublic'] = None
    created_at: datetime

class SequenceCreate(SequenceBase):
    ...

class SequenceUpdate(SQLModel):
    name: str | None = None
    duration: float | None = None
    gps: str | None = None
    labeled_frames: str | None = None


class SequenceUncertaintyBase(SQLModel):
    avg_uncertainty: float
    max_uncertainty: float
    uncertainty_per_frame: str

class SequenceUncertainty(SequenceUncertaintyBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sequence_id: int = Field(foreign_key='sequence.id')
    sequence: Sequence = Relationship(
        back_populates='uncertainty_measurements',
        sa_relationship_kwargs={
            "foreign_keys": "[SequenceUncertainty.sequence_id]"
        }
    )

class SequenceUncertaintyCreate(SequenceUncertaintyBase):
    sequence_name: str

class SequenceUncertaintyPublic(SequenceUncertaintyBase):
    id: int
    created_at: datetime


class DatasetBase(SQLModel):
    name: str = Field(index=True, unique=True)
    creator: str | None = Field(default=None)

class Dataset(DatasetBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    sequences: list[Sequence] = Relationship(back_populates='datasets', link_model=DatasetSequenceLink)
    training_runs: list['TrainingRun'] = Relationship(
        back_populates='dataset',
    )


class DatasetCreate(DatasetBase):
    train_sequence_names: list[str] = []
    val_sequence_names: list[str] = []

class DatasetUpdate(BaseModel):
    train_sequence_names_to_add: list[str] | None = None
    train_sequence_names_to_remove: list[str] | None = None
    val_sequence_names_to_add: list[str] | None = None
    val_sequence_names_to_remove: list[str] | None = None

class DatasetPublic(DatasetBase):
    id: int
    created_at: datetime


class TrainingStatus(str, Enum):
    NOT_STARTED = "NotStarted"
    STARTED = "Started"
    FINISHED = "Finished"
    FAILED = "Failed"

class TrainingRunBase(SQLModel):
    model_pt_path: str
    started_by: str | None = Field(default=None)
    # One of: 'NotStarted', 'Started', 'Finished', 'Failed'
    status: TrainingStatus = Field(default=TrainingStatus.NOT_STARTED)
    train_loss: float | None = Field(default=None)
    val_loss: float | None = Field(default=None)

class TrainingRun(TrainingRunBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    dataset_id: int | None = Field(default=None, foreign_key='dataset.id')
    dataset: Dataset | None = Relationship(back_populates='training_runs')

class TrainingRunCreate(TrainingRunBase):
    dataset_id: int

class TrainingRunUpdate(BaseModel):
    model_pt_path: str | None = None
    started_by: str | None = None
    status: TrainingStatus | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    dataset_id: int | None = None

class TrainingRunPublic(TrainingRunBase):
    id: int
    created_at: datetime


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def get_session():
    with Session(engine) as session:
        yield session

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("INFO:     Creating database and tables...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    create_db_and_tables()
    yield
    print("INFO:     Application shutting down.")


SessionDep = Annotated[Session, Depends(get_session)]
app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/testing/reset-database", status_code=200)
def reset_database():
    """
    Drops and recreates all tables for a clean test slate.
    This is a helper endpoint for the test suite.
    """
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    return {"message": "Database reset successfully."}


@app.post("/sequences/", response_model=SequencePublic)
def create_sequence(seq: SequenceCreate, session: SessionDep):
    db_seq = Sequence.model_validate(seq)
    session.add(db_seq)
    session.commit()
    session.refresh(db_seq)
    return db_seq


@app.get("/sequences/", response_model=list[SequencePublic])
def read_sequences(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=1000)] = 1000,
):
    return session.exec(select(Sequence).options(selectinload(Sequence.uncertainty_measurements)).offset(offset).limit(limit)).all()

@app.get("/sequences/{seq_id}", response_model=SequencePublic)
def read_sequence(seq_id: int, session: SessionDep):
    seq = session.exec(
        select(Sequence).options(selectinload(Sequence.uncertainty_measurements)).where(Sequence.id == seq_id)
    )
    if not seq:
        raise HTTPException(status_code=404, detail="Sequence not found")
    return seq

@app.patch("/sequences/{seq_id}", response_model=SequencePublic)
def update_sequence(seq_id: int, seq: SequenceUpdate, session: SessionDep):
    seq_db = session.get(Sequence, seq_id)
    if not seq_db:
        raise HTTPException(status_code=404, detail="Sequence not found")
    seq_data = seq.model_dump(exclude_unset=True)
    seq_db.sqlmodel_update(seq_data)
    session.add(seq_db)
    session.commit()
    session.refresh(seq_db)
    return seq_db

@app.delete("/sequences/{seq_id}")
def delete_sequence(seq_id: int, session: Session = Depends(get_session)):
    seq = session.get(Sequence, seq_id)
    if not seq:
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    session.delete(seq)
    session.commit()
    return

@app.post("/sequence_uncertainties/", response_model=SequenceUncertainty)
def create_sequence_uncertainty(seq_uncertainty: SequenceUncertaintyCreate, session: SessionDep):
    seq = session.exec(
        select(Sequence).where(Sequence.name == seq_uncertainty.sequence_name)
    ).first()

    if not seq:
        raise HTTPException(status_code=404, detail=f'Sequence with name {seq_uncertainty.sequence_name} doesnt exist.')

    db_uncertainty = SequenceUncertainty(
        **seq_uncertainty.model_dump(exclude={'sequence_name'}),
        sequence_id=seq.id
    )

    session.add(db_uncertainty)
    session.commit()
    session.refresh(db_uncertainty)
    return db_uncertainty

@app.get('/sequence_uncertainties', response_model=list[SequenceUncertaintyPublic])
def read_sequence_uncertainties(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=1000)] = 1000,
):
    return session.exec(select(SequenceUncertainty).offset(offset).limit(limit)).all()

@app.get('/sequence_uncertainties/{seq_unc_id}', response_model=SequenceUncertaintyPublic)
def read_sequence_uncertanty(seq_unc_id: int, session: SessionDep):
    seq_unc = session.get(SequenceUncertainty, seq_unc_id)
    if not seq_unc:
        raise HTTPException(status_code=404, detail='Sequence Uncertainty not found')
    return seq_unc

@app.delete("/sequence_uncertainties/{seq_id}")
def delete_sequence_uncertainties(seq_id: int, session: Session = Depends(get_session)):
    seq_unc = session.get(SequenceUncertainty, seq_id)
    if not seq_unc:
        raise HTTPException(status_code=404, detail="Sequence Uncertainty not found")
    
    session.delete(seq_unc)
    session.commit()
    return


def get_sequences_or_404(names: list[str], split_type: str, session: SessionDep) -> list[Sequence]:
    sequences = session.exec(select(Sequence).where(Sequence.name.in_(names))).all()
    if len(sequences) != len(names):
        missing = set(names) - {seq.name for seq in sequences}
        raise HTTPException(404, f"Missing {split_type} sequences: {', '.join(missing)}")
    return sequences

@app.post('/datasets/', response_model=DatasetPublic)
def create_dataset(ds_create: DatasetCreate, session: SessionDep):
    train_seqs = get_sequences_or_404(ds_create.train_sequence_names, "train", session)
    val_seqs = get_sequences_or_404(ds_create.val_sequence_names, "val", session)

    db_ds = Dataset(**ds_create.model_dump(exclude={'train_sequence_names', 'val_sequence_names'}))
    session.add(db_ds)
    session.flush()

    for seq, split in [(train_seqs, SplitType.TRAIN), (val_seqs, SplitType.VAL)]:
        session.add_all([DatasetSequenceLink(dataset_id=db_ds.id, sequence_id=s.id, split_type=split) for s in seq])

    session.commit()
    session.refresh(db_ds)
    return db_ds

@app.get('/datasets/{dataset_id}', response_model=DatasetPublic)
def read_dataset(dataset_id: int, session: SessionDep):
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail='Dataset not found')
    return dataset

@app.get('/datasets/', response_model=list[DatasetPublic])
def read_datasets(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=1000)] = 1000,
):
    return session.exec(select(Dataset).offset(offset).limit(limit)).all()

@app.patch('/datasets/{dataset_id}', response_model=DatasetPublic)
def update_dataset(dataset_id: int, update_data: DatasetUpdate, session: SessionDep):
    db_dataset = session.get(Dataset, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    def remove_links(names: list[str], split_type: SplitType):
        if not names:
            return
        sequences = get_sequences_or_404(names)
        session.exec(
            delete(DatasetSequenceLink).where(
                DatasetSequenceLink.dataset_id == dataset_id,
                DatasetSequenceLink.sequence_id.in_([seq.id for seq in sequences]),
                DatasetSequenceLink.split_type == split_type
            )
        )
    
    def add_links(names: list[str], split_type: SplitType):
        if not names:
            return
        sequences = get_sequences_or_404(names)
        existing = session.exec(
            select(DatasetSequenceLink.sequence_id).where(
                DatasetSequenceLink.dataset_id == dataset_id,
                DatasetSequenceLink.sequence_id.in_([seq.id for seq in sequences]),
                DatasetSequenceLink.split_type == split_type
            )
        ).all()

        new_links = [
            DatasetSequenceLink(dataset_id=dataset_id, sequence_id=seq.id, split_type=split_type)
            for seq in sequences if seq.id not in existing
        ]
        session.add_all(new_links)

    remove_links(update_data.train_sequence_names_to_remove or [], SplitType.TRAIN)
    remove_links(update_data.val_sequence_names_to_remove or [], SplitType.VAL)

    add_links(update_data.train_sequence_names_to_add or [], SplitType.TRAIN)
    add_links(update_data.val_sequence_names_to_add or [], SplitType.VAL)

    session.commit()
    session.refresh(db_dataset)
    return db_dataset

@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, session: Session = Depends(get_session)):
    ds = session.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    session.delete(ds)
    session.commit()
    return


@app.post("/training_runs/", response_model=TrainingRunPublic)
def create_training_run(
    training_run: TrainingRunCreate,
    session: Session = Depends(get_session)
):
    db_training_run = TrainingRun.model_validate(training_run)
    session.add(db_training_run)
    session.commit()
    session.refresh(db_training_run)
    return db_training_run

@app.get("/training_runs/", response_model=list[TrainingRunPublic])
def read_training_runs(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    # TODO: same for other read all calls
    status: Optional[TrainingStatus] = Query(None, description="Filter by training status"),
    started_by: Optional[str] = Query(None, description="Filter by started_by"),
    session: Session = Depends(get_session)
):
    query = select(TrainingRun)
    
    if status:
        query = query.where(TrainingRun.status == status)
    if started_by:
        query = query.where(TrainingRun.started_by == started_by)
    
    query = query.offset(skip).limit(limit)
    
    training_runs = session.exec(query).all()
    return training_runs

@app.get("/training_runs/{training_run_id}", response_model=TrainingRunPublic)
def read_training_run(training_run_id: int, session: Session = Depends(get_session)):
    training_run = session.get(TrainingRun, training_run_id)
    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return training_run

@app.patch("/training_runs/{training_run_id}", response_model=TrainingRunPublic)
def update_training_run(
    training_run_id: int,
    training_run_update: TrainingRunUpdate,
    session: Session = Depends(get_session)
):
    db_training_run = session.get(TrainingRun, training_run_id)
    if not db_training_run:
        raise HTTPException(status_code=404, detail="Training run not found")

    update_data = training_run_update.model_dump(exclude_unset=True)
    db_training_run.sqlmodel_update(update_data)

    session.add(db_training_run)
    session.commit()
    session.refresh(db_training_run)
    return db_training_run

@app.delete("/training_runs/{training_run_id}")
def delete_training_run(training_run_id: int, session: Session = Depends(get_session)):
    training_run = session.get(TrainingRun, training_run_id)
    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    session.delete(training_run)
    session.commit()
    return


@app.get('/sequence_maps/{seq_id}')
def get_sequence_map(seq_id: int, session: SessionDep):
    seq_db = session.get(Sequence, seq_id)
    if not seq_db:
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    gps_hash = xxhash.xxh64(seq_db.gps).hexdigest()

    img_path = f'{CACHE_DIR}/osm_{seq_id}_{gps_hash}.png'
    if not os.path.exists(img_path):
        _create_and_save_image(json.loads(seq_db.gps), img_path)

    return FileResponse(img_path, media_type='image/png')
