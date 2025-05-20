"""
cloud_launcher.py

Google Colab Launcher for QMP Overrider

Provides integration with Google Colab for cloud-based backtesting and analysis.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta

class ColabLauncher:
    """
    Google Colab Launcher for QMP Overrider
    
    Provides integration with Google Colab for cloud-based backtesting and analysis.
    """
    
    def __init__(self):
        """Initialize the Google Colab Launcher"""
        self.config = self._load_config()
        self.notebook_templates = self._load_templates()
        self.last_launch_time = None
        self.launch_history = []
        self.max_history = 100
        self.initialized = False
    
    def _load_config(self):
        """
        Load configuration
        
        Returns:
        - Dictionary with configuration
        """
        config = {
            "symbols": ["BTCUSD", "ETHUSD", "XAUUSD", "DIA", "QQQ"],
            "backtest_period": 90,  # days
            "default_cash": 100000,
            "default_leverage": 1.0,
            "notebook_path": "notebook_templates",
            "data_path": "data",
            "results_path": "results",
            "github_repo": "https://github.com/username/QMP_Overrider_QuantConnect",
            "colab_url": "https://colab.research.google.com/github/username/QMP_Overrider_QuantConnect/blob/main/notebooks/"
        }
        
        config_path = os.path.join(os.path.dirname(__file__), "colab_config.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    loaded_config = json.load(f)
                    
                    for key, value in loaded_config.items():
                        config[key] = value
            except Exception as e:
                print(f"Error loading configuration: {e}")
        
        return config
    
    def _load_templates(self):
        """
        Load notebook templates
        
        Returns:
        - Dictionary with notebook templates
        """
        templates = {
            "backtest": None,
            "analysis": None,
            "optimization": None,
            "live_trading": None,
            "dashboard": None
        }
        
        template_dir = os.path.join(os.path.dirname(__file__), self.config["notebook_path"])
        
        if os.path.exists(template_dir):
            for template_name in templates.keys():
                template_path = os.path.join(template_dir, f"{template_name}.ipynb")
                
                if os.path.exists(template_path):
                    try:
                        with open(template_path, "r") as f:
                            templates[template_name] = f.read()
                    except Exception as e:
                        print(f"Error loading template {template_name}: {e}")
        
        return templates
    
    def initialize(self):
        """
        Initialize the Google Colab Launcher
        
        Returns:
        - True if successful, False otherwise
        """
        if self.initialized:
            return True
        
        self._create_directories()
        
        self._create_default_templates()
        
        self.initialized = True
        
        print("Google Colab Launcher: Initialized")
        
        return True
    
    def _create_directories(self):
        """Create required directories"""
        template_dir = os.path.join(os.path.dirname(__file__), self.config["notebook_path"])
        os.makedirs(template_dir, exist_ok=True)
        
        data_dir = os.path.join(os.path.dirname(__file__), self.config["data_path"])
        os.makedirs(data_dir, exist_ok=True)
        
        results_dir = os.path.join(os.path.dirname(__file__), self.config["results_path"])
        os.makedirs(results_dir, exist_ok=True)
    
    def _create_default_templates(self):
        """Create default notebook templates"""
        pass
    
    def launch_notebook(self, template_name, params=None):
        """
        Launch a notebook
        
        Parameters:
        - template_name: Name of the notebook template to launch
        - params: Dictionary with parameters to pass to the notebook (optional)
        
        Returns:
        - URL of the launched notebook
        """
        if not self.initialized:
            self.initialize()
        
        if template_name not in self.notebook_templates or self.notebook_templates[template_name] is None:
            print(f"Template {template_name} not found")
            return None
        
        notebook = self._create_notebook(template_name, params)
        
        notebook_path = self._save_notebook(template_name, notebook)
        
        url = self._get_notebook_url(template_name)
        
        self._record_launch(template_name, params, url)
        
        return url
    
    def _create_notebook(self, template_name, params=None):
        """
        Create a notebook from a template
        
        Parameters:
        - template_name: Name of the notebook template
        - params: Dictionary with parameters to pass to the notebook (optional)
        
        Returns:
        - Notebook as JSON string
        """
        template = self.notebook_templates[template_name]
        
        try:
            notebook = json.loads(template)
        except Exception as e:
            print(f"Error parsing template {template_name}: {e}")
            return None
        
        if params:
            params_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Parameters\n",
                    "params = " + json.dumps(params, indent=4)
                ]
            }
            
            for i, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "markdown":
                    notebook["cells"].insert(i + 1, params_cell)
                    break
        
        return json.dumps(notebook, indent=2)
    
    def _save_notebook(self, template_name, notebook):
        """
        Save a notebook
        
        Parameters:
        - template_name: Name of the notebook template
        - notebook: Notebook as JSON string
        
        Returns:
        - Path to the saved notebook
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{template_name}_{timestamp}.ipynb"
        
        notebook_path = os.path.join(
            os.path.dirname(__file__),
            self.config["notebook_path"],
            filename
        )
        
        try:
            with open(notebook_path, "w") as f:
                f.write(notebook)
        except Exception as e:
            print(f"Error saving notebook {filename}: {e}")
            return None
        
        return notebook_path
    
    def _get_notebook_url(self, template_name):
        """
        Get URL for a notebook
        
        Parameters:
        - template_name: Name of the notebook template
        
        Returns:
        - URL of the notebook
        """
        base_url = self.config["colab_url"]
        
        url = f"{base_url}{template_name}.ipynb"
        
        return url
    
    def _record_launch(self, template_name, params, url):
        """
        Record a notebook launch
        
        Parameters:
        - template_name: Name of the notebook template
        - params: Dictionary with parameters passed to the notebook
        - url: URL of the launched notebook
        """
        launch = {
            "template": template_name,
            "params": params,
            "url": url,
            "timestamp": datetime.now()
        }
        
        self.launch_history.append(launch)
        
        self.last_launch_time = launch["timestamp"]
        
        if len(self.launch_history) > self.max_history:
            self.launch_history = self.launch_history[-self.max_history:]
    
    def get_launch_history(self, max_count=None):
        """
        Get launch history
        
        Parameters:
        - max_count: Maximum number of records to return (optional)
        
        Returns:
        - List of launch records
        """
        if max_count:
            return self.launch_history[-max_count:]
        
        return self.launch_history
    
    def get_status(self):
        """
        Get Google Colab Launcher status
        
        Returns:
        - Dictionary with status information
        """
        return {
            "initialized": self.initialized,
            "templates": list(self.notebook_templates.keys()),
            "last_launch_time": self.last_launch_time,
            "launch_count": len(self.launch_history)
        }
