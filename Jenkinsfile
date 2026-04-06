@Library('groovylibrary') _

pipeline {
    agent { label 'docker-agent' }

    environment {
        IMAGE_NAME = 'pytorch-gpu'
        REGISTRY   = 'registry.example.com'
        NAMESPACE  = 'gpu-workloads'
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 20, unit: 'MINUTES')
        timestamps()
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build & Push Image') {
            when { branch 'main' }
            steps {
                dockerBuildPush(
                    imageName: env.IMAGE_NAME,
                    registry: env.REGISTRY,
                    credentialsId: 'docker-registry-creds',
                    dockerfile: 'Dockerfile'
                )
            }
        }

        stage('Deploy to K8s') {
            when { branch 'main' }
            steps {
                deployK8s(
                    manifestPath: 'pytorch-gpu-deployment.yaml',
                    namespace: env.NAMESPACE,
                    kubeConfigCredential: 'kubeconfig-gpu',
                    deploymentName: 'pytorch-gpu-deployment',
                    timeoutMinutes: 5
                )
            }
        }
    }

    post {
        failure {
            llmAnalyzeFailure(
                apiCredential: 'openai-api-key',
                slackChannel: '#ml-infra',
                webhookCredential: 'slack-webhook-url'
            )
        }
        always {
            cleanWs()
        }
    }
}
