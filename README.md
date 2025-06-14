# 📄 README.md

## Agriculture Analytics Platform

A comprehensive machine learning-based analytics platform designed to predict agricultural yields and optimize resourc## 📬 Contact Contributors

For support or questions, please contact Victor Busami at [victorbusami1@gmail.com](mailto:victorbusami1@gmail.com).sage. This project is available both as a Jupyter Notebook for data analysis and as an interactive Flask web application.

## About

Click the link to get our pitch deck link:

🔗 [gamma.app/docs/Revolutionizing-Agriculture-with-Data-Driven-Insights](https://gamma.app/docs/Revolutionizing-Agriculture-with-Data-Driven-Insights-g0s8al76rdggogq)

---

## 📂 Project Structure

### Jupyter Notebook Analysis

- `agric_analysis_training.ipynb`: The main notebook containing all analysis steps including:
  - Data Loading: Uses `agriculture_dataset.csv` (should be in the same directory) with crop yield, season, and input data.
  - Exploratory Data Analysis (EDA): Visualizations (histograms, boxplots, scatter plots) to understand variable distributions and detect outliers.
  - Data Preprocessing: Handles missing values, applies scaling, and encodes categorical variables.
  - Modeling: Trains both Linear Regression and Random Forest Regressor models.
  - Model Evaluation: Uses metrics such as RMSE and R² score to evaluate performance.
  - Insights: Summarizes findings including feature importance and seasonal impacts.

### Flask Web Application

- `app.py`: Main Flask application file
- `app/`: Directory containing web application components:

  - `templates/`: HTML templates for the web interface
  - `static/`: CSS, JavaScript, and image files
  - `models/`: Trained machine learning models and metrics

- `agriculture_dataset.csv`: The dataset containing agricultural yield data

---

## 🚀 Getting Started

### 📥 Clone the Repository

```bash
git clone https://github.com/victor-busami/revolutionizing-agriculture-webapp.git
cd revolutionizing-agriculture-webapp
```

### 🛠️ Install Dependencies

The project requires Python 3.x and the following libraries:

- flask
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- plotly
- flask-wtf
- joblib
- gunicorn (for deployment)

To install all dependencies:

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 How to Run the Project

### Option 1: Run the Flask Web Application

1. Make sure you have activated your virtual environment (if using one).

2. Start the Flask application:

   ```bash
   python app.py
   ```

3. Open a web browser and navigate to: http://127.0.0.1:5000

4. Explore the different features of the application:
   - Browse the dataset on the Data page
   - View visualizations and analytics on the Visualizations page
   - Access the dashboard for key metrics and insights
   - Use the prediction tool to estimate crop yields based on various parameters

### Option 2: Run the Jupyter Notebook

1. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Open `agric_analysis_training.ipynb`.
3. Run each cell sequentially:

   - Load the dataset.
   - Perform EDA to understand data distributions and patterns.
   - Preprocess the data (handle missing values, encoding, scaling).
   - Train and evaluate models.
   - Review the results including RMSE and R² scores.

---

## 🔍 Key Features and Functionality

### Data Analysis Features

- **Data Exploration:** Understand seasonal trends, identify outliers, and visualize variable relationships.
- **Data Preprocessing:** Cleans data and prepares it for modeling (e.g., using OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler).
- **Modeling:** Compares multiple regression models (Linear Regression and Random Forest Regressor) for yield prediction.
- **Model Evaluation:** Includes RMSE and R² metrics with insights on model performance.
- **Feature Importance:** Highlights key variables that influence crop yield.
- **Seasonal Analysis:** Explores agricultural seasons (Kharif, Rabi, Zaid) and their impact on yield.

### Web Application Features

- **Interactive Dashboard:** Key metrics and insights displayed in an easy-to-understand format
- **Data Browser:** View and explore the agricultural dataset
- **Visualization Tools:** Interactive charts and graphs to analyze agricultural patterns
- **Yield Prediction:** Input your parameters to get crop yield predictions
- **Responsive Design:** Access the application on desktop and mobile devices

---

## 🌱 Supporting the Sustainable Development Goals (SDGs)

This project aligns with:

- **SDG 2: Zero Hunger** — Enhancing agricultural productivity through predictive analytics.
- **SDG 12: Responsible Consumption and Production** — Optimizing resource use with data-driven insights.
- **SDG 13: Climate Action** — Supporting climate-resilient agricultural practices.
- **SDG 8: Decent Work and Economic Growth** — Creating opportunities for agritech innovation.

---

## 🤝 Contributing

We welcome contributions! Suggestions, bug reports, and pull requests are appreciated.

---

## � Deployment

To deploy this application to a production server:

1. Create a Procfile for platforms like Heroku:

   ```
   web: gunicorn app:app
   ```

2. Set appropriate environment variables for production.

3. Deploy to your preferred hosting platform (Heroku, AWS, DigitalOcean, etc.)

### 🌐 Deployment

#### Deploying to Render

This application is configured for easy deployment on Render:

1. Create a free account on [Render](https://render.com/)
2. Connect your GitHub repository
3. Create a new Web Service
4. Use the following settings:
   - Name: agricultural-analytics
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Choose the free plan
5. Click "Create Web Service"

The application includes a `render.yaml` configuration file to simplify deployment.

---

## �📬 Contact Contributors

For support or questions, please contact Victor Busami at [[victor.busami@example.com](mailto:victor.busami@example.com)].

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.

# Team Collaboration Guide - Please Read Before Contributing!

## Hey Team! 👋

Welcome to our project! I've put together this guide to help everyone collaborate smoothly on our GitHub repository. Please take a few minutes to read through this - it'll save us all time and prevent headaches down the road.

## What You'll Need Before Starting

- Git installed on your machine (if you don't have it, grab it from [git-scm.com](https://git-scm.com))
- Access to our GitHub repository (let me know if you need an invite)
- Basic Git knowledge (don't worry, I'll walk you through everything)
- **Python and Jupyter Notebook** installed. You can install using:

```bash
pip install notebook
```

---

## Working With Jupyter Notebooks (Without VS Code)

Since we’re using **Jupyter Notebooks** in this project, here’s how you can work with them directly from your terminal or command prompt (no need for VS Code):

### ✅ After Cloning the Repository:

1. **Navigate into the project folder:**

```bash
cd repository-name
```

2. **Check if the Jupyter notebook is present:**

```bash
ls *.ipynb  # On Linux/Mac
# or
dir *.ipynb  # On Windows
```

If it’s missing or not listed, let the team know.

3. **Launch the Jupyter Notebook interface:**

```bash
jupyter notebook
```

This will open a new browser tab showing the notebook dashboard.

4. **Open the `.ipynb` file:**

   - Click on the notebook file to open it in your browser.
   - Make your changes carefully and test your cells.

5. **Save your work:**

   - Click `File > Save and Checkpoint`, or simply press `Ctrl + S`

6. **Close the notebook and stop the server:**

   - Once done, close the notebook tab.
   - Stop the Jupyter server by pressing `Ctrl + C` in your terminal.

7. **Stage and commit your changes:**

```bash
git add filename.ipynb
git commit -m "Update notebook with [your changes]"
```

8. **Push your branch to GitHub:**

```bash
git push origin your-branch-name
```

> ✅ **Avoid merge conflicts:**
>
> - Don’t edit the same notebook cell as someone else.
> - Always pull from `main` before editing.

---

## Our Team Workflow - Please Follow These Steps!

### Step 1: Get the Repository on Your Machine (First Time Only)

```bash
git clone https://github.com/your-username/repository-name.git
cd repository-name
```

### Step 2: ALWAYS Update Main First! (This is Super Important!)

```bash
git checkout main
git pull origin main
```

### Step 3: Create Your Own Branch

```bash
git checkout -b feature/your-awesome-feature
```

**Branch naming tips:**

- `feature/description`
- `bugfix/what-you-fixed`
- `hotfix/urgent-fix`
- `docs/what-you-updated`

### Step 4: Do Your Magic! ✨

- Edit your `.ipynb` file or Python code.
- Run cells to test and make sure everything works.

### Step 5: Save Your Work

```bash
git status
git add .
git commit -m "Add amazing feature that does X and Y"
```

### Step 6: Push Your Branch

```bash
git push origin feature/your-awesome-feature
```

### Step 7: Ask for a Code Review (Pull Request)

- Use GitHub to open a Pull Request.
- Fill in title, description, screenshots, and testing steps.

### Step 8: Work With Me on Reviews

```bash
git add .
git commit -m "Fix review changes"
git push origin feature/your-awesome-feature
```

### Step 9: Celebrate and Clean Up 🎉

```bash
git checkout main
git pull origin main
git branch -d feature/your-awesome-feature
git push origin --delete feature/your-awesome-feature
```

---

## Team Rules - Please Respect These!

### ❌ DON'T:

- Push directly to `main`
- Use `--force` on shared branches
- Commit secrets
- Upload huge files without asking

### ✅ DO:

- Work on branches
- Pull main before edits
- Write clear commits
- Test code
- Keep PRs focused

---

## Useful Git Commands:

```bash
git status
git branch -a
git checkout branch-name
git log --oneline
git reset --soft HEAD~1
```

### Merge Conflict Help:

```bash
git checkout main
git pull origin main
git checkout your-branch
git merge main
# Fix conflicts, then:
git add .
git commit -m "Resolve merge conflicts"
git push origin your-branch
```

### If Using a Fork:

```bash
git remote add upstream https://github.com/original-owner/repository-name.git
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

---

## Quick Git + Notebook Cheat Sheet

```bash
git checkout main
git pull origin main
git checkout -b feature/new-thing
# Launch Jupyter Notebook, edit your .ipynb
jupyter notebook
# Save and exit notebook
# Then:
git add your-notebook.ipynb
git commit -m "What you did"
git push origin feature/new-thing
```

## Need Help? Just Ask!

- Ping me directly
- Create a GitHub issue
- Ask in the group chat

## Final Thoughts

Thanks for reading this! Let’s keep things clean, simple, and helpful. Follow the workflow, support each other, and let's build something awesome 🚀
