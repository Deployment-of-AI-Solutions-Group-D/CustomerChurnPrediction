pipeline {
    agent any
    
    environment {
        // Environment variable for the Flask application name
        APP_NAME = 'CustomerChurnPrediction'
    }

    stages {
        stage('Clone Repository') {
            steps {
                // Clone the Git repository from the specified branch
                git url: 'https://github.com/Deployment-of-AI-Solutions-Group-D/CustomerChurnPrediction.git', branch: 'jini_zacharias'
            }
        }
        
        stage('Install Dependencies') {
            steps {
                script {
                    // Check the operating system and run appropriate commands
                    if (isUnix()) {
                        // On Unix/Linux systems, set up a virtual environment and install dependencies
                        sh '''
                        python3 --version
                        pip3 --version
                        python3 -m venv venv
                        . venv/bin/activate
                        pip install -r requirements.txt
                        '''
                    } else {
                        // On Windows systems, set up a virtual environment and install dependencies
                        bat '''
                        python --version
                        pip --version
                        python -m venv venv
                        call venv\\Scripts\\activate && pip install -r requirements.txt
                        '''
                    }
                }
            }
        }
        
        stage('Run Unit Tests') {
            steps {
                script {
                    // Run unit tests based on the operating system
                    if (isUnix()) {
                        sh '''
                        . venv/bin/activate
                        venv/bin/pytest tests/ || { echo "pytest failed"; exit 1; }
                        '''
                    } else {
                        bat '''
                        call venv\\Scripts\\activate && venv\\Scripts\\pytest tests\\ || (echo pytest failed && exit /b 1)
                        '''
                    }
                }
            }
        }
        
        stage('Start Application') {
            steps {
                script {
                    // Start the Flask application in the background
                    if (isUnix()) {
                        // Use nohup for Unix/Linux
                        sh '''
                        nohup python app.py &
                        '''
                    } else {
                        // Use start /B for Windows
                        bat '''
                        start /B python app.py
                        '''
                    }
                }
            }
        }
        
        stage('Test Application') {
            steps {
                script {
                    // Wait for a few seconds to ensure the app starts before testing
                    sleep(5)
                    
                    // You can use curl or a simple Python script to test if the app is running
                    if (isUnix()) {
                        sh '''
                        curl http://localhost:80/ui || echo "Application is not responding"
                        '''
                    } else {
                        bat '''
                        curl http://localhost:80/ui || echo Application is not responding
                        '''
                    }
                }
            }
        }
        
        stage('Cleanup') {
            steps {
                script {
                    // Optionally clean up resources or stop the application if necessary
                    if (isUnix()) {
                        sh 'pkill -f app.py || true' // Stop the Flask app
                    } else {
                        bat 'taskkill /F /IM python.exe || echo No running instance of app.py to kill'
                    }
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
