# LinkedIn Post — Wine Quality Prediction with XGBoost and SageMaker HPO

Transformei um projeto de modelagem em um case end-to-end de Machine Learning Engineering.

O objetivo: prever a qualidade de vinhos a partir de propriedades físico-químicas, usando XGBoost, AWS SageMaker e uma estrutura preparada para MLOps.

O fluxo construído cobre:

1. Ingestão de dados de vinhos branco e tinto
2. Análise exploratória, correlação e detecção de outliers
3. Pré-processamento e preparação dos dados
4. Split treino / validação / teste
5. Treinamento com XGBoost
6. Otimização de hiperparâmetros com SageMaker HPO
7. Avaliação com MSE, MAE e RMSE
8. Organização de artefatos e documentação técnica
9. Estrutura inicial para deploy via API

Resultado do modelo XGBoost otimizado:

- MSE: 0.4564
- MAE: 0.3867
- RMSE: 0.6756

O ponto principal deste projeto não foi apenas treinar um modelo.

Foi estruturar o caminho completo:

dados → features → tuning → modelo → avaliação → artefatos → deploy

Esse tipo de fluxo é o que aproxima um notebook experimental de uma solução real de Machine Learning.

Em produção, o modelo é só uma parte do sistema.

O diferencial está em construir uma solução rastreável, reprodutível, documentada e pronta para evoluir.

Repositório completo no GitHub:
[adicione o link aqui]

#MachineLearning #DataScience #MLOps #XGBoost #AWS #SageMaker #Python #Portfolio
