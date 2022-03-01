using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>
    /// Нейросеть
    /// </summary>
    class Network
    {
        struct LayerT
        {
            public Vector x; // Вход слоя
            public Vector z; // Активированный выход слоя
            public Vector df; // Производная функции активации слоя
        }

        Matrix[] weights; // Матрицы весов слоя
        LayerT[] L; // Значения на каждом слое
        Vector[] deltas; // Дельты ошибки на каждом слое

        int layersN; // Число слоёв

        /// <summary>
        /// Создание сети из массива количества нейронов в каждом слое
        /// </summary>
        /// <param name="sizes">Размер</param>
        public Network(int[] sizes)
        {
            Random random = new Random(DateTime.Now.Millisecond); // Создаём генератор случайных чисел

            layersN = sizes.Length - 1; // Запоминаем число слоёв

            weights = new Matrix[layersN]; // Создаём массив матриц весовых коэффициентов
            L = new LayerT[layersN]; // Создаём массив значений на каждом слое
            deltas = new Vector[layersN]; // Создаём массив для дельт

            // Создаём матрицы весовых коэффициентов для каждого слоя
            for (int k = 1; k < sizes.Length; k++)
            {
                weights[k - 1] = new Matrix(sizes[k], sizes[k - 1], random); // Создаём матрицу весовых коэффициентов

                L[k - 1].x = new Vector(sizes[k - 1]); // Создаём вектор для входа слоя
                L[k - 1].z = new Vector(sizes[k]); // Создаём вектор для выхода слоя
                L[k - 1].df = new Vector(sizes[k]); // Создаём вектор для производной слоя

                deltas[k - 1] = new Vector(sizes[k]); // Создаём вектор для дельт
            }
        }

        /// <summary>
        /// Получение выхода сети (прямое распространение)
        /// </summary>
        /// <param name="input">Входной вектор</param>
        /// <returns>Результат</returns>
        public Vector Forward(Vector input)
        {
            for (int k = 0; k < layersN; k++)
            {
                if (k == 0)
                {
                    for (int i = 0; i < input.n; i++)
                        L[k].x[i] = input[i];
                }
                else
                {
                    for (int i = 0; i < L[k - 1].z.n; i++)
                        L[k].x[i] = L[k - 1].z[i];
                }

                for (int i = 0; i < weights[k].n; i++)
                {
                    double y = 0; // Неактивированный выход нейрона

                    for (int j = 0; j < weights[k].m; j++)
                        y += weights[k][i, j] * L[k].x[j];

                    // Выполняем активацию с помощью сигмоидальной функциии
                    L[k].z[i] = 1 / (1 + Math.Exp(-y));
                    L[k].df[i] = L[k].z[i] * (1 - L[k].z[i]);

                    // Выполняем активацию с помощью гиперболического тангенса
                    //L[k].z[i] = Math.Tanh(y);
                    //L[k].df[i] = 1 - L[k].z[i] * L[k].z[i];

                    // Выполняем активацию с помощью ReLU
                    //L[k].z[i] = y > 0 ? y : 0;
                    //L[k].df[i] = y > 0 ? 1 : 0;
                }
            }

            return L[layersN - 1].z; // Возвращаем результат
        }

        /// <summary>
        /// Обратное распространение
        /// </summary>
        /// <param name="output">Выход</param>
        /// <param name="error">Ошибка</param>
        void Backward(Vector output, ref double error)
        {
            int last = layersN - 1;

            error = 0; // Обнуляем ошибку

            for (int i = 0; i < output.n; i++)
            {
                double e = L[last].z[i] - output[i]; // Находим разность значений векторов

                deltas[last][i] = e * L[last].df[i]; // Запоминаем дельту
                error += e * e / 2; // Прибавляем к ошибке половину квадрата значения
            }

            // Вычисляем каждую предудущю дельту на основе текущей с помощью умножения на транспонированную матрицу
            for (int k = last; k > 0; k--)
            {
                for (int i = 0; i < weights[k].m; i++)
                {
                    deltas[k - 1][i] = 0;

                    for (int j = 0; j < weights[k].n; j++)
                        deltas[k - 1][i] += weights[k][j, i] * deltas[k][j];

                    deltas[k - 1][i] *= L[k - 1].df[i]; // Умножаем получаемое значение на производную предыдущего слоя
                }
            }
        }

        /// <summary>
        /// Обновление весовых коэффициентов
        /// </summary>
        /// <param name="alpha">Cкорость обучения</param>
        void UpdateWeights(double alpha)
        {
            for (int k = 0; k < layersN; k++)
            {
                for (int i = 0; i < weights[k].n; i++)
                {
                    for (int j = 0; j < weights[k].m; j++)
                    {
                        weights[k][i, j] -= alpha * deltas[k][i] * L[k].x[j];
                    }
                }
            }
        }

        /// <summary>
        /// Обучение сети
        /// </summary>
        /// <param name="X">Массив входных обучающих векторов</param>
        /// <param name="Y">Массив выходных обучающих векторов</param>
        /// <param name="alpha">Скорость обучения</param>
        /// <param name="eps"></param>
        /// <param name="epochs"></param>
        public void Train(Vector[] X, Vector[] Y, double alpha, double eps, int epochs)
        {
            int epoch = 1; // Номер эпохи

            double error; // Ошибка эпохи

            do
            {
                error = 0; // Обнуляем ошибку

                // Проходимся по всем элементам обучающего множества
                for (int i = 0; i < X.Length; i++)
                {
                    Forward(X[i]); // Прямое распространение сигнала
                    Backward(Y[i], ref error); // Обратное распространение ошибки
                    UpdateWeights(alpha); // Обновление весовых коэффициентов
                }

                Console.WriteLine($"epoch: {epoch}, error: {error}"); // Выводим в консоль номер эпохи и величину ошибку

                epoch++; // Увеличиваем номер эпохи
            } while (epoch <= epochs && error > eps);
        }
    }
}
