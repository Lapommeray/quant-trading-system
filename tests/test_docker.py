import pytest
import subprocess
try:
    import docker
except ImportError:
    docker = None
    pytest.skip("docker not available", allow_module_level=True)
from pathlib import Path

class TestDocker:
    
    def test_dockerfile_exists(self):
        dockerfile = Path("Dockerfile")
        assert dockerfile.exists(), "Dockerfile not found"
        
        with open(dockerfile, 'r') as f:
            content = f.read()
        
        assert "FROM python:" in content, "Dockerfile missing Python base image"
        assert "WORKDIR" in content, "Dockerfile missing WORKDIR"
        assert "COPY requirements.txt" in content, "Dockerfile missing requirements copy"
    
    def test_dockerignore_exists(self):
        dockerignore = Path(".dockerignore")
        assert dockerignore.exists(), ".dockerignore not found"
        
        with open(dockerignore, 'r') as f:
            content = f.read()
        
        assert ".git" in content, ".dockerignore missing .git"
        assert "__pycache__" in content, ".dockerignore missing __pycache__"
        assert "*.pyc" in content, ".dockerignore missing *.pyc"
    
    def test_docker_compose_services(self):
        compose_file = Path("docker-compose.yml")
        assert compose_file.exists(), "docker-compose.yml not found"
        
        import yaml
        with open(compose_file, 'r') as f:
            compose = yaml.safe_load(f)
        
        required_services = [
            "qts-api",
            "qts-dashboard", 
            "postgres",
            "redis",
            "prometheus",
            "grafana"
        ]
        
        for service in required_services:
            assert service in compose['services'], f"Missing service: {service}"
    
    @pytest.mark.slow
    def test_docker_build(self):
        if docker is None:
            pytest.skip("Docker not available")
        try:
            client = docker.from_env()
        except Exception:
            pytest.skip("Docker not available")
        
        try:
            result = subprocess.run(
                ["docker", "build", "-t", "qts-test", "."],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            assert result.returncode == 0, f"Docker build failed: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Docker build timed out")
        except FileNotFoundError:
            pytest.skip("Docker command not found")
    
    @pytest.mark.slow
    def test_docker_compose_validate(self):
        try:
            result = subprocess.run(
                ["docker-compose", "config"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0, f"docker-compose config invalid: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.fail("docker-compose config timed out")
        except FileNotFoundError:
            pytest.skip("docker-compose command not found")
