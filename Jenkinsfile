pipeline {
    agent any
    
    stages {
        stage('Clone Repository') {
            steps {
                // Check out the repository from version control
                git url: 'https://github.com/Deployment-of-AI-Solutions-Group-D/CustomerChurnPrediction.git', branch: 'master'
            }
        }
        
        stage('Install Dependencies') {
            steps {
                // Install the dependencies required for your Flask app
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Run Unit Tests') {
            steps {
                // Run your tests (assuming you have unit tests for your app)
                sh 'pytest tests/'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                // Build a Docker image for your Flask app
                sh 'docker build -t your-flask-app:latest .'
            }
        }
        
        stage('Push Docker Image') {
            steps {
                // Push the Docker image to a Docker registry (e.g., DockerHub)
                withCredentials([usernamePassword(credentialsId: 'docker-credentials-id', passwordVariable: 'DOCKER_PASSWORD', usernameVariable: 'DOCKER_USERNAME')]) {
                    sh 'echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin'
                    sh 'docker tag your-flask-app:latest your-dockerhub-username/your-flask-app:latest'
                    sh 'docker push your-dockerhub-username/your-flask-app:latest'
                }
            }
        }
        
        stage('Deploy') {
            steps {
                // Deploy the application using Docker (assuming you're using Docker for deployment)
                sshagent(['your-ssh-credentials-id']) {
                    sh 'ssh user@your-server "docker pull your-dockerhub-username/your-flask-app:latest && docker stop flask_app || true && docker rm flask_app || true && docker run -d --name flask_app -p 80:80 your-dockerhub-username/your-flask-app:latest"'
                }
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed.'
        }
    }
}
