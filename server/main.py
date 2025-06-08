from typing import Annotated, Optional
import json
import os
from datetime import datetime
from enum import Enum

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from contextlib import asynccontextmanager
import xxhash

from map import _create_and_save_image

CACHE_DIR = '.cache'


class DatasetSequenceLink(SQLModel, table=True):
    sequence_id: int | None = Field(default=None, foreign_key='sequence.id', primary_key=True)
    dataset_id: int | None = Field(default=None, foreign_key='dataset.id', primary_key=True)


class SequenceBase(SQLModel):
    name: str = Field(index=True, unique=True)
    duration: float
    gps: str | None = Field(default=None)

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
    sequence_names: list[str]

class DatasetUpdate(BaseModel):
    sequence_names_to_add: list[str]
    sequence_names_to_remove: list[str]

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
    creator: str | None = Field(default=None)
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
    creator: str | None = None
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


@app.post('/datasets/', response_model=DatasetPublic)
def create_dataset(ds_create: DatasetCreate, session: SessionDep):
    sequences_in_db = session.exec(
        select(Sequence).where(Sequence.name.in_(ds_create.sequence_names))
    ).all()

    if len(sequences_in_db) != len(ds_create.sequence_names):
        found_names = {seq.name for seq in sequences_in_db}
        missing_names = set(ds_create.sequence_names) - found_names
        raise HTTPException(
            status_code=404,
            detail=f"The following sequences were not found: {', '.join(missing_names)}"
        )
    
    db_ds = Dataset(**ds_create.model_dump(exclude={'sequence_names'}))
    db_ds.sequences = sequences_in_db

    session.add(db_ds)
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
    
    current_sequence_names = {seq.name for seq in db_dataset.sequences}
    
    if update_data.sequence_names_to_remove:
        names_to_remove_set = set(update_data.sequence_names_to_remove)
        db_dataset.sequences = [
            seq for seq in db_dataset.sequences 
            if seq.name not in names_to_remove_set
        ]
        current_sequence_names -= names_to_remove_set
    
    if update_data.sequence_names_to_add:
        new_names_to_add = [
            name for name in update_data.sequence_names_to_add 
            if name not in current_sequence_names
        ]
        if new_names_to_add:
            sequences_from_db = session.exec(
                select(Sequence).where(Sequence.name.in_(new_names_to_add))
            ).all()

            if len(sequences_from_db) != len(new_names_to_add):
                found_names = {seq.name for seq in sequences_from_db}
                missing_names = set(new_names_to_add) - found_names
                raise HTTPException(
                    status_code=404,
                    detail=f"Cannot add sequences, the following were not found: {', '.join(missing_names)}"
                ) 

        db_dataset.sequences.extend(sequences_from_db)
    
    session.add(db_dataset)
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
    creator: Optional[str] = Query(None, description="Filter by creator"),
    session: Session = Depends(get_session)
):
    query = select(TrainingRun)
    
    if status:
        query = query.where(TrainingRun.status == status)
    if creator:
        query = query.where(TrainingRun.creator == creator)
    
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
