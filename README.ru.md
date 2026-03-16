# Глава 361: Оператор Купмана в Трейдинге — Линеаризация Нелинейной Динамики Рынка

## Обзор

Финансовые рынки — это по своей природе нелинейные динамические системы. Традиционные линейные методы с трудом справляются со сложными паттернами, сменами режимов и нестационарным поведением. **Оператор Купмана** предлагает революционный подход: он преобразует нелинейную динамику в **линейную** (но бесконечномерную) систему через функции-наблюдаемые.

В этой главе мы исследуем, как использовать теорию оператора Купмана и её алгоритмические приближения (Динамическая Декомпозиция Мод, Расширенная DMD, Глубокие Сети Купмана) для торговых приложений, включая прогнозирование, определение режимов и извлечение признаков.

## Торговая Стратегия

**Основная концепция:** Использование методов оператора Купмана для:
1. **Линеаризации** сложной рыночной динамики в расширенном пространстве признаков
2. **Извлечения когерентных мод** (динамических мод), описывающих эволюцию цены
3. **Прогнозирования будущих состояний** с использованием изученной линейной динамики
4. **Обнаружения смены режимов** через спектральный анализ оператора Купмана

**Преимущество:** В то время как другие моделируют сырые ценовые ряды нелинейными методами, подходы Купмана находят внутреннюю линейную структуру, скрытую в данных, обеспечивая более интерпретируемые и стабильные прогнозы.

## Математические Основы

### Оператор Купмана

Для дискретной динамической системы:
```
x_{t+1} = F(x_t)
```

Оператор Купмана K действует на **наблюдаемые** (скалярные функции состояния):
```
(K g)(x) = g(F(x))
```

Ключевой инсайт: K — **линейный**, даже когда F нелинейна!

### Разложение по Собственным Функциям

Если φ — собственная функция K с собственным значением λ:
```
K φ = λ φ
```

Тогда:
```
φ(x_t) = λ^t φ(x_0)
```

Это означает, что **любая наблюдаемая** может быть разложена на моды с известной временной эволюцией.

### Динамическая Декомпозиция Мод (DMD)

DMD аппроксимирует оператор Купмана из снимков данных:

Даны матрицы данных:
```
X = [x_1, x_2, ..., x_{m-1}]
Y = [x_2, x_3, ..., x_m]
```

Найти линейный оператор A такой, что Y ≈ AX:
```
A = Y X^†  (псевдообратная)
```

Моды DMD — это собственные векторы A, а собственные значения дают скорости роста и частоты.

### Расширенная DMD (EDMD)

Поднимаем данные в пространство более высокой размерности с использованием словарных функций:
```
Ψ(x) = [ψ_1(x), ψ_2(x), ..., ψ_N(x)]^T
```

Затем применяем DMD к поднятым данным:
```
Ψ(Y) ≈ K̂ Ψ(X)
```

### Глубокие Сети Купмана

Используем нейронные сети для обучения оптимальных функций подъёма:
```
Энкодер: x → Ψ(x)
Слой Купмана: Ψ(x_t) → K Ψ(x_t) ≈ Ψ(x_{t+1})
Декодер: Ψ(x) → x
```

Функции потерь:
1. **Потеря реконструкции**: ||x - Decoder(Encoder(x))||²
2. **Потеря прогноза**: ||x_{t+1} - Decoder(K · Encoder(x_t))||²
3. **Потеря линейности**: ||Encoder(x_{t+1}) - K · Encoder(x_t)||²

## Техническая Спецификация

### Структура Модулей

```
361_koopman_operator_trading/
├── README.md                 # Основная глава (английский)
├── README.ru.md              # Русский перевод
├── readme.simple.md          # Простое объяснение (английский)
├── readme.simple.ru.md       # Простое объяснение (русский)
├── README.specify.md         # Спецификация
└── rust/
    ├── Cargo.toml
    └── src/
        ├── lib.rs            # Корень библиотеки
        ├── main.rs           # Точка входа CLI
        ├── api/
        │   ├── mod.rs
        │   └── bybit.rs      # Клиент API Bybit
        ├── koopman/
        │   ├── mod.rs
        │   ├── dmd.rs        # Динамическая Декомпозиция Мод
        │   ├── edmd.rs       # Расширенная DMD
        │   ├── kernels.rs    # Ядерные функции для EDMD
        │   └── prediction.rs # Утилиты прогнозирования
        ├── features/
        │   ├── mod.rs
        │   ├── observables.rs # Функции-наблюдаемые
        │   └── lifting.rs     # Подъём признаков
        ├── trading/
        │   ├── mod.rs
        │   ├── signals.rs    # Генерация торговых сигналов
        │   ├── backtest.rs   # Движок бэктестинга
        │   └── metrics.rs    # Метрики производительности
        └── data/
            ├── mod.rs
            └── types.rs      # Структуры данных
```

### Основные Алгоритмы

#### 1. Стандартная DMD

```rust
/// Динамическая Декомпозиция Мод
pub struct DMD {
    pub modes: Array2<Complex64>,       // Моды DMD (собственные векторы)
    pub eigenvalues: Array1<Complex64>, // Собственные значения Купмана
    pub amplitudes: Array1<Complex64>,  // Амплитуды мод
    pub dt: f64,                        // Шаг времени
}

impl DMD {
    pub fn fit(data: &Array2<f64>, dt: f64) -> Result<Self> {
        let (n, m) = data.dim();

        // Разделение на матрицы X и Y
        let x = data.slice(s![.., ..m-1]).to_owned();
        let y = data.slice(s![.., 1..]).to_owned();

        // SVD матрицы X
        let (u, s, vt) = svd(&x)?;

        // Усечение по рангу
        let r = optimal_rank(&s);
        let u_r = u.slice(s![.., ..r]);
        let s_r = Array::from_diag(&s.slice(s![..r]));
        let vt_r = vt.slice(s![..r, ..]);

        // Построение редуцированной матрицы
        let a_tilde = u_r.t().dot(&y).dot(&vt_r.t()).dot(&s_r.inv());

        // Разложение по собственным значениям
        let (eigenvalues, eigenvectors) = eig(&a_tilde)?;

        // Восстановление полных DMD мод
        let modes = y.dot(&vt_r.t()).dot(&s_r.inv()).dot(&eigenvectors);

        // Вычисление амплитуд
        let amplitudes = lstsq(&modes, &data.column(0))?;

        Ok(Self {
            modes,
            eigenvalues,
            amplitudes,
            dt,
        })
    }

    pub fn predict(&self, steps: usize, x0: &Array1<f64>) -> Array2<f64> {
        let mut predictions = Array2::zeros((x0.len(), steps));

        for t in 0..steps {
            let time_dynamics: Array1<Complex64> = self.eigenvalues
                .iter()
                .zip(self.amplitudes.iter())
                .map(|(λ, b)| b * λ.powf(t as f64))
                .collect();

            let pred = self.modes.dot(&time_dynamics);
            predictions.column_mut(t).assign(&pred.mapv(|c| c.re));
        }

        predictions
    }

    pub fn continuous_eigenvalues(&self) -> Array1<Complex64> {
        self.eigenvalues.mapv(|λ| λ.ln() / self.dt)
    }

    pub fn frequencies(&self) -> Array1<f64> {
        self.continuous_eigenvalues().mapv(|ω| ω.im / (2.0 * PI))
    }

    pub fn growth_rates(&self) -> Array1<f64> {
        self.continuous_eigenvalues().mapv(|ω| ω.re)
    }
}
```

#### 2. Расширенная DMD со Словарём

```rust
/// Словарные функции для EDMD
pub trait Dictionary: Send + Sync {
    fn lift(&self, x: &Array1<f64>) -> Array1<f64>;
    fn dim(&self) -> usize;
}

/// Полиномиальный словарь
pub struct PolynomialDictionary {
    pub degree: usize,
    pub input_dim: usize,
}

impl Dictionary for PolynomialDictionary {
    fn lift(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut features = vec![1.0]; // Константный член

        // Добавление мономов до заданной степени
        for d in 1..=self.degree {
            for combo in combinations_with_replacement(0..self.input_dim, d) {
                let term: f64 = combo.iter().map(|&i| x[i]).product();
                features.push(term);
            }
        }

        Array1::from_vec(features)
    }

    fn dim(&self) -> usize {
        binomial(self.input_dim + self.degree, self.degree)
    }
}

/// Словарь RBF (Радиальных Базисных Функций)
pub struct RBFDictionary {
    pub centers: Array2<f64>,
    pub sigma: f64,
}

impl Dictionary for RBFDictionary {
    fn lift(&self, x: &Array1<f64>) -> Array1<f64> {
        let n_centers = self.centers.nrows();
        let mut features = Array1::zeros(n_centers + 1);
        features[0] = 1.0; // Константный член

        for (i, center) in self.centers.rows().into_iter().enumerate() {
            let diff = x - &center.to_owned();
            let dist_sq = diff.dot(&diff);
            features[i + 1] = (-dist_sq / (2.0 * self.sigma.powi(2))).exp();
        }

        features
    }

    fn dim(&self) -> usize {
        self.centers.nrows() + 1
    }
}

/// Расширенная Динамическая Декомпозиция Мод
pub struct EDMD<D: Dictionary> {
    pub dictionary: D,
    pub koopman_matrix: Array2<f64>,
    pub eigenvalues: Array1<Complex64>,
    pub eigenvectors: Array2<Complex64>,
}
```

#### 3. Генерация Торговых Сигналов

```rust
/// Торговые сигналы на основе Купмана
pub struct KoopmanTrader {
    pub dmd: DMD,
    pub prediction_horizon: usize,
    pub threshold: f64,
}

impl KoopmanTrader {
    pub fn generate_signal(&self, current_state: &Array1<f64>) -> Signal {
        // Прогноз будущих цен
        let predictions = self.dmd.predict(self.prediction_horizon, current_state);
        let current_price = current_state[0];
        let predicted_price = predictions[[0, self.prediction_horizon - 1]];

        // Расчёт ожидаемой доходности
        let expected_return = (predicted_price - current_price) / current_price;

        // Анализ стабильности мод
        let stable_modes = self.dmd.growth_rates()
            .iter()
            .filter(|&&r| r < 0.0)
            .count();
        let stability_ratio = stable_modes as f64 / self.dmd.eigenvalues.len() as f64;

        // Генерация сигнала с уверенностью
        let confidence = stability_ratio * (1.0 - self.prediction_uncertainty());

        if expected_return > self.threshold && confidence > 0.5 {
            Signal::Long {
                strength: expected_return.min(1.0),
                confidence,
            }
        } else if expected_return < -self.threshold && confidence > 0.5 {
            Signal::Short {
                strength: (-expected_return).min(1.0),
                confidence,
            }
        } else {
            Signal::Neutral
        }
    }

    /// Обнаружение смены режима через спектральный анализ
    pub fn detect_regime_change(&self, window1: &Array2<f64>, window2: &Array2<f64>) -> f64 {
        let dmd1 = DMD::fit(window1, self.dmd.dt).unwrap();
        let dmd2 = DMD::fit(window2, self.dmd.dt).unwrap();

        // Сравнение распределений собственных значений
        spectral_distance(&dmd1.eigenvalues, &dmd2.eigenvalues)
    }
}
```

### Наблюдаемые Функции для Финансов

```rust
/// Финансовые наблюдаемые для анализа Купмана
pub struct FinancialObservables {
    pub lookback: usize,
}

impl FinancialObservables {
    pub fn compute(&self, prices: &[f64]) -> Array1<f64> {
        let n = prices.len();
        let mut observables = Vec::new();

        // Сама цена
        observables.push(prices[n-1]);

        // Логарифмические доходности
        for lag in 1..=self.lookback.min(5) {
            if n > lag {
                let log_ret = (prices[n-1] / prices[n-1-lag]).ln();
                observables.push(log_ret);
            }
        }

        // Скользящие средние
        for window in [5, 10, 20].iter() {
            if n >= *window {
                let ma: f64 = prices[n-window..].iter().sum::<f64>() / *window as f64;
                observables.push(ma);
            }
        }

        // Волатильность
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        if returns.len() >= 10 {
            let recent_returns = &returns[returns.len()-10..];
            let mean = recent_returns.iter().sum::<f64>() / 10.0;
            let vol = (recent_returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / 10.0).sqrt();
            observables.push(vol);
        }

        // Моментум
        if n >= 10 {
            let momentum = prices[n-1] / prices[n-10] - 1.0;
            observables.push(momentum);
        }

        Array1::from_vec(observables)
    }
}
```

## Торговые Стратегии

### Стратегия 1: Прогнозирование по Модам DMD

Использование доминирующих мод DMD для краткосрочного прогноза цен.

### Стратегия 2: Определение Режимов

Обнаружение смены режимов путём мониторинга спектра Купмана.

### Стратегия 3: Мультиактивный Купман

Анализ кросс-активной динамики с выявлением lead-lag отношений.

## Ключевые Метрики

### Качество Модели
- **Ошибка реконструкции**: ||X - X_reconstructed||
- **RMSE прогноза**: Среднеквадратичная ошибка прогнозов
- **Стабильность мод**: Доля собственных значений внутри единичного круга

### Торговая Производительность
- **Коэффициент Шарпа**: Доходность с учётом риска
- **Коэффициент Сортино**: Доходность с учётом нижнего риска
- **Максимальная просадка**: Наибольшее падение от пика до дна
- **Процент выигрышей**: Доля прибыльных сделок

### Спектральные Метрики
- **Спектральный зазор**: Разделение между доминирующей и вторичными модами
- **Когерентность**: Насколько хорошо моды объясняют дисперсию
- **Частотное содержание**: Доминирующие частоты колебаний

## Зависимости

```toml
[dependencies]
ndarray = { version = "0.15", features = ["serde"] }
ndarray-linalg = { version = "0.16", features = ["openblas-static"] }
num-complex = "0.4"
reqwest = { version = "0.11", features = ["json"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }
anyhow = "1.0"
thiserror = "1.0"
```

## Примечания по Реализации

### Численная Стабильность

1. **Усечение SVD**: Всегда используйте усечённое SVD для матриц с недостаточным рангом
2. **Регуляризация**: Добавляйте малую диагональную регуляризацию для предотвращения сингулярных матриц
3. **Нормализация**: Нормализуйте данные перед DMD для улучшения численной стабильности

### Практические Соображения

1. **Вложение с задержкой**: Используйте координаты задержки для преобразования скалярного временного ряда в пространство состояний
2. **Выбор окна**: Баланс между захватом динамики (большое окно) и стационарностью (малое окно)
3. **Выбор мод**: Фильтруйте моды по стабильности, энергии и релевантности
4. **Онлайн-обновления**: Рассмотрите инкрементальную DMD для потоковых приложений

## Ожидаемые Результаты

1. **Реализация DMD** с возможностями прогнозирования
2. **EDMD с различными словарями** (полиномиальный, RBF, пользовательский)
3. **Генератор торговых сигналов** на основе прогнозов Купмана
4. **Определение режимов** через спектральный анализ
5. **Мультиактивный анализ**, включая lead-lag отношения
6. **Фреймворк бэктестинга** для стратегий Купмана

## Литература

1. **Deep Learning for Universal Linear Embeddings of Nonlinear Dynamics**
   - URL: https://arxiv.org/abs/1712.09707
   - Ключевой вклад: Глубокие автоэнкодеры Купмана

2. **Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems**
   - Авторы: Kutz, Brunton, Brunton, Proctor
   - Комплексный учебник по DMD

3. **Data-Driven Science and Engineering**
   - Авторы: Brunton & Kutz
   - Глава по DMD и оператору Купмана

4. **Extended Dynamic Mode Decomposition with Dictionary Learning**
   - URL: https://arxiv.org/abs/1510.04765
   - Теоретические основы EDMD

## Уровень Сложности

**Экспертный** (5/5)

Необходимые знания:
- Линейная алгебра (SVD, разложение по собственным значениям)
- Теория динамических систем
- Анализ временных рядов
- Программирование на Rust

Эта глава сочетает продвинутые математические концепции с практическими торговыми приложениями.
