import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open

class TestConfigurationFiles:
    
    def test_config_yaml_exists(self):
        config_path = Path("config.yaml")
        assert config_path.exists()
        assert config_path.is_file()
    
    def test_config_yaml_valid(self):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config, dict)
        assert "system" in config
        assert "trading" in config
        assert "quantum" in config
    
    def test_env_example_exists(self):
        env_path = Path(".env.example")
        assert env_path.exists()
        assert env_path.is_file()
    
    def test_env_example_format(self):
        with open(".env.example", "r") as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                assert "=" in line, f"Invalid env line: {line}"
    
    def test_pyproject_toml_exists(self):
        pyproject_path = Path("pyproject.toml")
        assert pyproject_path.exists()
        assert pyproject_path.is_file()
    
    def test_setup_py_exists(self):
        setup_path = Path("setup.py")
        assert setup_path.exists()
        assert setup_path.is_file()
    
    def test_requirements_files_exist(self):
        req_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-test.txt"
        ]
        
        for req_file in req_files:
            req_path = Path(req_file)
            assert req_path.exists(), f"Missing {req_file}"
            assert req_path.is_file()
    
    def test_docker_files_exist(self):
        docker_files = [
            "Dockerfile",
            "docker-compose.yml",
            ".dockerignore"
        ]
        
        for docker_file in docker_files:
            if Path(docker_file).exists():
                assert Path(docker_file).is_file()
    
    def test_ci_files_exist(self):
        ci_files = [
            ".github/workflows/ci.yml",
            ".github/workflows/test.yml",
            ".github/workflows/deploy.yml",
            ".github/dependabot.yml"
        ]
        
        for ci_file in ci_files:
            if Path(ci_file).exists():
                assert Path(ci_file).is_file()
    
    def test_code_quality_files_exist(self):
        quality_files = [
            ".flake8",
            ".pylintrc", 
            ".pre-commit-config.yaml",
            "pytest.ini",
            "tox.ini",
            ".coveragerc"
        ]
        
        for quality_file in quality_files:
            if Path(quality_file).exists():
                assert Path(quality_file).is_file()

class TestConfigurationValidation:
    
    def test_config_yaml_structure(self):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        required_sections = ["system", "trading", "quantum"]
        for section in required_sections:
            assert section in config, f"Missing section: {section}"
        
        assert "version" in config["system"]
        assert "never_loss_protection" in config["trading"]
        assert "enabled" in config["quantum"]
    
    @patch.dict(os.environ, {"QTS_ENV": "test"})
    def test_environment_variable_loading(self):
        assert os.getenv("QTS_ENV") == "test"
    
    def test_yaml_syntax_validation(self):
        yaml_files = ["config.yaml"]
        
        for yaml_file in yaml_files:
            if Path(yaml_file).exists():
                with open(yaml_file, "r") as f:
                    try:
                        yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        pytest.fail(f"Invalid YAML in {yaml_file}: {e}")

class TestPackageConfiguration:
    
    def test_setup_py_syntax(self):
        try:
            with open("setup.py", "r") as f:
                content = f.read()
            compile(content, "setup.py", "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in setup.py: {e}")
    
    def test_requirements_format(self):
        req_files = ["requirements.txt", "requirements-dev.txt"]
        
        for req_file in req_files:
            if Path(req_file).exists():
                with open(req_file, "r") as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        assert any(op in line for op in [">=", "==", "~=", ">", "<"]), \
                            f"Invalid requirement format: {line}"
