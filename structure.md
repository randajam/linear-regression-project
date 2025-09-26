01-linear-regression/
│
├── README.md              # краткое описание проекта
├── plan.md                # полный учебный план (чеклист выполнения)
├── requirements.txt       # библиотеки
│
├── notebooks/
│   ├── 00_questions.ipynb         # теория (ответы на 1 раздел)
│   ├── 01_preprocessing.ipynb     # вводные шаги: загрузка, обработка interest_level
│   ├── 02_features.ipynb          # фича-инжиниринг (20 топ-фич + bedrooms/bathrooms)
│   ├── 03_linear.ipynb            # реализация линейной регрессии (analytical, GD, SGD)
│   ├── 04_regularization.ipynb    # Ridge, Lasso, ElasticNet
│   ├── 05_normalization.ipynb     # MinMaxScaler, StandardScaler
│   ├── 06_poly_overfit.ipynb      # полиномы, оверфит, регуляризация
│   ├── 07_naive.ipynb             # наивные модели (mean, median)
│   ├── 08_compare.ipynb           # сравнение моделей, таблицы метрик
│   └── 09_additional.ipynb        # дополнительные задания (лог-таргет, выбросы, mini-batch)
│
├── src/
│   ├── __init__.py
│   ├── linear.py         # классы для линейной регрессии (analytical, GD, SGD)
│   ├── regularized.py    # Ridge, Lasso, ElasticNet
│   ├── scalers.py        # MinMaxScaler, StandardScaler
│   ├── metrics.py        # R2, MAE, RMSE
│   └── utils.py          # вспомогательные функции
│
├── datasets/
│   └── README.md         # где скачать и как подготовить
│
└── results/
    ├── metrics.csv
    └── plots/
