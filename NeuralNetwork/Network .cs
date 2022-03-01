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
    }
}
