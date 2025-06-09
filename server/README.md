# ML Dataset Managment

Install
~~~
python3 -m venv venv
pip install -r requirements.txt
~~~

Run
~~~
fastapi dev main.py --host 0.0.0.0
python create_test_data.py
~~~


TODO

- somehow reference files used -> can be inferred from db data
- handle missing gps, undefined


### **Platform Name: Lidar Analytics & Model Management (LAMM)**

**Core Philosophy:** To provide a centralized, versioned, and visual platform for the entire lifecycle of point cloud data, from raw ingestion to model training and evaluation. It aims to bridge the gap between data curation, annotation, and machine learning experimentation.

---

### **Component 1: Main Dashboard**

This is the landing page after a user logs in. It provides a high-level, at-a-glance overview of the entire system.

*   **Key Stats Widgets:**
    *   **Total Scenes:** Count of all scenes in the database.
    *   **Labeling Status:** Pie chart showing (Unlabeled, In Progress, Labeled, In Review).
    *   **Data Volume:** Total storage used by PCD files (e.g., 5.2 TB).
    *   **Active ML Runs:** Number of training experiments currently in the "Running" state.
*   **Recent Activity Feed:**
    *   List of recent events, e.g., "User A uploaded 50 new scenes," "Training Run #124 completed," "Scene 'urban-drive-07' was labeled."
*   **Quick Access Links:**
    *   "Upload New Scenes"
    *   "Create New Dataset"
    *   "Start New Training Run"
    *   "View Best Performing Model"

---

### **Component 2: Data Explorer (Scene Library)**

This is the primary interface for browsing, searching, and managing all the raw and labeled point cloud scenes.

*   **View:** A powerful, filterable table view.

*   **Table Columns:**
    *   **Thumbnail:** A pre-generated 2D image preview of the point cloud.
    *   **Scene ID / Name:** A unique, human-readable identifier (e.g., `2023-10-26-highway-drive-01`).
    *   **Duration (s):** The length of the data capture in seconds.
    *   **GPS Coordinates:** Start/End latitude and longitude. Could be a link to an external map view.
    *   **Point Count:** Total number of points in the cloud (e.g., 15.4M).
    *   **Date Captured:** Timestamp of the data acquisition.
    *   **Labeling Status:** A status tag (e.g., `Unlabeled`, `In Progress`, `Labeled`).
    *   **Labeler:** User who performed the labeling.
    *   **Used in Training:** A boolean flag (`Yes`/`No`) or a list of Training Run IDs it was part of.
    *   **Tags:** User-defined, searchable tags (e.g., `urban`, `night`, `rain`, `occlusion`, `challenging`).

*   **Functionality:**
    *   **Advanced Filtering:** Filter by any column, especially by `Tags` and `Labeling Status`.
    *   **Bulk Actions:** Select multiple scenes to:
        *   Add tags.
        *   Assign to a labeler.
        *   Add to a new or existing Dataset.
    *   **Search Bar:** Full-text search on Scene ID and tags.

---

### **Component 3: Scene Detail View**

This view opens when a user clicks on a specific scene in the Data Explorer.

*   **Header:** Scene ID, key metadata, and status.
*   **Interactive 3D Viewer:**
    *   A large panel for visualizing the point cloud.
    *   Controls for pan, zoom, rotate.
    *   **Overlay Toggles:** Buttons to show/hide:
        *   Raw Point Cloud
        *   Ground Truth Labels (if available), colored by class (e.g., Ground=blue, Obstacle=red).
        *   Model Prediction Overlay (User can select a model from the Model Registry to see its predictions on this scene).
*   **Metadata Panel:**
    *   All information from the Data Explorer table in a detailed list.
    *   File path to the raw PCD file.
    *   File path to the label file.
*   **History & Associations:**
    *   **Labeling History:** Audit log (e.g., "Assigned to User B on...", "Labeled by User B on...").
    *   **Dataset Membership:** Lists which datasets include this scene.
    *   **Training Run History:** Lists all training runs that used this scene (in either training or validation sets).

---

### **Component 4: Dataset Builder**

This component allows users to group scenes into versioned datasets for reproducible training.

*   **Interface:** A two-panel view.
    *   **Left Panel:** A replica of the Data Explorer to find and select scenes.
    *   **Right Panel:** The "Dataset Staging Area."
*   **Workflow:**
    1.  User gives the dataset a name (e.g., `Urban_Night_Scenes_v1`).
    2.  User filters and selects scenes from the left panel and adds them to the right.
    3.  User defines data splits (e.g., using sliders or percentage inputs for Training, Validation, and Test sets).
    4.  The platform can show statistics for the created dataset (e.g., "Total scenes: 200, Train: 160, Val: 20, Test: 20").
    5.  User saves the dataset. It is now immutable and versioned.

---

### **Component 5: Experiment Tracker (ML Runs Overview)**

This is the dashboard for all training jobs, fulfilling a core requirement.

*   **View:** A filterable table of all ML experiments.

*   **Table Columns:**
    *   **Run ID:** A unique ID for the training job (e.g., `run-124`).
    *   **Status:** `Queued`, `Running`, `Completed`, `Failed`.
    *   **Model Architecture:** The type of model used (e.g., `PointNet++`, `RandLA-Net`).
    *   **Training Loss:** The final training loss value.
    *   **Validation Loss:** The final validation loss value.
    *   **Validation IoU (Obstacle):** Intersection-over-Union for the "obstacle" class. A critical metric.
    *   **Validation IoU (Ground):** IoU for the "ground" class.
    *   **Dataset Used:** Link to the versioned dataset from the Dataset Builder.
    *   **Model Path:** Link to the stored model artifact.
    *   **Started By:** The user who initiated the run.
    *   **Duration:** How long the training took.

---

### **Component 6: Run Detail View**

This view opens when a user clicks on a specific run in the Experiment Tracker.

*   **Header:** Run ID, Status, Final Metrics.
*   **Performance Plots:** Interactive charts showing metrics over training epochs/steps:
    *   Training Loss vs. Validation Loss.
    *   Validation IoU (per class) over time.
    *   Learning Rate schedule.
*   **Configuration Panel:**
    *   A read-only view of all hyperparameters used for the run (learning rate, batch size, optimizer, number of epochs, etc.).
    *   The exact code/commit hash used for the training script, ensuring full reproducibility.
*   **Evaluation & Visualization Tab:**
    *   **Validation Set Performance:** A table of all scenes in the validation set, showing the model's IoU on each one. This helps identify where the model struggles.
    *   **Visual Comparison Tool:** A side-by-side or overlay view in the 3D viewer for any validation scene, comparing **Ground Truth** vs. **Model Prediction**. Users can quickly cycle through the best and worst-performing scenes.
    *   **Confusion Matrix:** A visual confusion matrix for the entire validation set.

---

### **Component 7: Model Registry**

A centralized repository for trained models that have been "promoted" from an experiment.

*   **View:** Table of all registered models.
*   **Information for each model:**
    *   **Model Name & Version:** e.g., `production-classifier-v3.2`.
    *   **Source Run:** A link back to the Experiment Tracker run that produced it.
    *   **Key Metrics:** A summary of its performance (IoU, loss).
    *   **Deployment Status:** Tags like `Development`, `Staging`, `Production`.
    *   **Release Notes:** A description of what changed or improved in this version.






We'll treat "Ensemble Uncertainty" as a first-class citizen, just like "Loss" or "IoU." The data you have (uncertainty per frame/point cloud) will be associated with a specific Model Ensemble Evaluation Run.
Integration Strategy
The core idea is to:
Store the uncertainty score for each frame alongside a reference to the model ensemble that produced it.
Surface this information in data-centric views to help find problematic scenes.
Analyze this information in model-centric views to understand model weaknesses.
Create a dedicated overview to directly answer the user's primary question.
Modified Component 2: Data Explorer (Scene Library)
This view is about discovering data. We'll add uncertainty metrics to help users find scenes that models find "confusing."
New Table Columns:
Max Frame Uncertainty: The single highest uncertainty score recorded for any frame within that scene, across all evaluated models. This immediately flags scenes with at least one very difficult moment.
Avg. Frame Uncertainty: The average uncertainty score across all frames in that scene. This indicates if a scene is generally challenging.
Uncertainty Source Model: The ID of the model/ensemble whose evaluation produced the "Max Frame Uncertainty" score. This tells you which model struggled the most.
Enhanced Functionality:
Sorting: Users can now sort the entire scene library by "Max Frame Uncertainty" in descending order. This is the primary way to get a ranked list of the most poorly performing scenes.
Filtering: Users can filter for scenes with "Max Frame Uncertainty > [threshold]" to create datasets specifically for re-training on hard cases.
Modified Component 3: Scene Detail View
When a user clicks on a scene, they need to see where and why it has high uncertainty.
New Visualization Panel: Uncertainty Timeline
A new interactive plot displayed prominently in this view.
X-axis: Time (in seconds) or Frame Number of the scene.
Y-axis: Ensemble Uncertainty Score.
Functionality:
The plot shows the uncertainty score for each frame, revealing spikes of high uncertainty.
Interactive Brushing: The user can click on a peak in the timeline. This action will:
Instantly load that specific frame (point cloud) into the 3D viewer.
Trigger the "Uncertainty Overlay" (see below).
Enhanced 3D Viewer:
New Overlay Toggle: A button called "Show Uncertainty Overlay."
Functionality: When toggled on, the points in the cloud are re-colored based on their individual uncertainty scores (if you have per-point uncertainty) or a uniform color representing the frame's overall uncertainty.
Color Mapping: Use an intuitive heat map (e.g., Cool Blue for low uncertainty -> Hot Red for high uncertainty). This immediately draws the user's eye to the specific objects or areas the model is struggling to classify.
Modified Component 6: Run Detail View
When analyzing a specific model run, we need to know how certain or uncertain that model was.
New Summary Metrics (in Header):
Mean Validation Uncertainty: The average uncertainty score across all frames in the validation set for this specific model.
95th Percentile Uncertainty: The uncertainty value below which 95% of the validation frames fall. Good for understanding the "worst-case" behavior.
New Tab: "Uncertainty Analysis"
Uncertainty Distribution Histogram: A chart showing how many validation frames fall into different uncertainty buckets (e.g., 0-0.1, 0.1-0.2, ...). This shows if the model is generally confident or has a wide spread of uncertainty.
Top 10 Most Uncertain Frames: A table listing the 10 frames from the validation set where this model exhibited the highest uncertainty.
Columns: Frame ID, Scene ID, Uncertainty Score, IoU (to see if high uncertainty correlates with low IoU).
Links: Each row links directly to the Scene Detail View, pre-loading that specific frame and its uncertainty overlay.
New Component 8: Uncertainty Hotspot Dashboard
This new, dedicated dashboard directly provides the "overview of poorly performing scenes" that you requested. It aggregates uncertainty information across all models and all data.
View 1: Global Scene Ranking
A table of all scenes in the database, ranked by their "Max Frame Uncertainty" score (the highest score they've ever received from any model evaluation).
Purpose: To find scenes that are universally difficult, regardless of the model. These are prime candidates for re-labeling, data augmentation, or detailed review.
Columns: Scene ID, Max Uncertainty Score, Model that produced this score, Date, Tags.
View 2: Model-Centric Analysis
User Input: A dropdown to select a specific Model from the Model Registry.
Display: The dashboard updates to show a ranked list of scenes where that specific model was most uncertain.
Purpose: To debug a particular model and understand its unique failure modes.
View 3: Tag-Based Uncertainty Breakdown
A bar chart or box-plot visualization.
X-axis: Data Tags (e.g., rain, night, urban, occlusion).
Y-axis: Average Uncertainty Score.
Purpose: To answer questions like, "Is our ensemble generally more uncertain at night or in the rain?" This provides high-level, actionable insights into systemic weaknesses and guides future data collection efforts.
Example Insight: If the rain tag has a significantly higher average uncertainty, the team knows they need to collect more rainy-day data or develop rain-specific augmentation techniques.