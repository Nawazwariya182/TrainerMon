pipeline {

    agent any



    environment {

        PYTHONUNBUFFERED = '1'

        TRAINERMON_API = 'http://localhost:3000'

        ACCURACY_THRESHOLD = '0.85'

    }



    options {

        timeout(time: 60, unit: 'MINUTES')


    }



    stages {

        stage('Checkout SCM') {

            steps {

                echo "=========================================================="

                echo "  TRAINERMON CI/CD: Automated Pipeline Execution"

                echo "=========================================================="

                echo "Workspace: ${WORKSPACE}"

            }

        }



        stage('Verify Environment') {

            steps {

                echo "[Stage 0/5] Verifying Python Environment & Dependencies..."

                bat '''

                    python --version

                    pip install -r requirements.txt

                '''

            }

        }



        stage('Download Datasets') {

            steps {

                echo "[Stage 1/5] Ingesting Multi-Dataset Sources..."

                bat '''

                    python download_datasets.py

                '''

            }

        }



        stage('EDA & Tensor Harmonization') {

            steps {

                echo "[Stage 2/5] Running Exploratory Data Analysis & Tensor Harmonization..."

                bat '''

                    python eda_and_harmonization.py

                '''

            }

        }



        stage('Pretraining Loop') {

            steps {

                echo "[Stage 3/5] Executing Neural Pretraining Loop from Scratch..."

                bat '''

                    python train.py

                '''

            }

        }



        stage('Evaluation & Quality Gate') {

            steps {

                echo "[Stage 4/5] Running Benchmark Evaluation & Scoring..."

                bat '''

                    python evaluate_and_score.py

                '''

                script {

                    echo "Checking Accuracy Quality Gate (Target >= 85%)..."

                    def metricsFile = "metrics.json"

                    if (fileExists(metricsFile)) {

                        def props = readJSON file: metricsFile

                        def acc = props.accuracy ?: props.val_accuracy ?: 0.0

                        echo "Evaluated Accuracy: ${acc}"

                        if (acc < 0.85) {

                            error("QUALITY_GATE_FAILED: Achieved accuracy ${acc} is below required threshold 0.85.")

                        } else {

                            echo "QUALITY_GATE_PASSED: Achieved accuracy ${acc} satisfies >= 0.85 threshold!"

                        }

                    }

                }

            }

        }



        stage('Generate Visualizations') {

            steps {

                echo "[Stage 5/5] Rendering Overlays & Loss Curves..."

                bat '''

                    python visualize.py

                '''

            }

        }

    }



    post {

        always {

            archiveArtifacts artifacts: 'metrics.json, EXECUTION_STATUS.json, config.yaml, visualizations/*.png', allowEmptyArchive: true

        }

        success {

            powershell '''

                try {

                    $body = @{ event = "PIPELINE_SUCCESS"; buildNumber = $env:BUILD_NUMBER; jobName = $env:JOB_NAME } | ConvertTo-Json

                    Invoke-RestMethod -Uri "http://localhost:3000/api/ci/webhook" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 15

                } catch {}

            '''

        }

        failure {

            powershell '''

                try {

                    $body = @{ event = "PIPELINE_FAILURE"; buildNumber = $env:BUILD_NUMBER; jobName = $env:JOB_NAME } | ConvertTo-Json

                    Invoke-RestMethod -Uri "http://localhost:3000/api/ci/webhook" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 15

                } catch {}

            '''

        }

    }

}