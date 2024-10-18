pipeline {
    agent any
    
    environment {
        // Environment variable for the Flask application name
        APP_NAME = 'CustomerChurnPrediction'
    }

    stages {
        stage('Clone Repository') {
            steps {
                // Clone the Git repository
                git url: 'https://github.com/Deployment-of-AI-Solutions-Group-D/CustomerChurnPrediction.git', branch: 'master'
            }
        }
        
        stage('Install Dependencies') {
            steps {
                script {
                    // Check the operating system and run appropriate commands
                    if (isUnix()) {
                        // On Unix/Linux systems, set up a virtual environment and install dependencies
                        sh '''
                        python3 -m venv venv
                        . venv/bin/activate
                        pip install -r requirements.txt
                        '''
                    } else {
                        // On Windows systems, set up a virtual environment and install dependencies
                        bat '''
                        python -m venv venv
                        venv\\Scripts\\activate
                        pip install -r requirements.txt
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
                        sh '. venv/bin/activate && pytest tests/'
                    } else {
                        bat 'venv\\Scripts\\activate && pytest tests\\'
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
        
        stage('Cleanup') {
            steps {
                script {
                    // Optionally clean up resources or stop the application if necessary
                    if (isUnix()) {
                        sh 'pkill -f app.py || true' // Stop the Flask app
                    } else {
                        bat 'taskkill /F /IM python.exe || echo "No running instance of app.py to kill"'
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
