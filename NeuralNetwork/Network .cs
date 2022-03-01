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
        Matrix[] weights; // Матрицы весов слоя

        int layersN; // Число слоёв

        /// <summary>
        /// Создание сети из массива количества нейронов в каждом слое
        /// </summary>
        /// <param name="sizes">Размер</param>
        public Network(int[] sizes)
        {
            Random random = new Random(DateTime.Now.Millisecond); // Создаём генератор случайных чисел

            layersN = sizes.Length - 1; // Запоминаем число слоёв

            weights = new Matrix[layersN]; // Создаём массив матриц

            // Создаём матрицы весовых коэффициентов для каждого слоя
            for (int k = 1; k < sizes.Length; k++)
            {
                weights[k - 1] = new Matrix(sizes[k], sizes[k - 1], random);
            }
        }

        /// <summary>
        /// Получение выхода сети (прямое распространение)
        /// </summary>
        /// <param name="input">Входной вектор</param>
        /// <returns>Выходной вектор</returns>
        Vector Forward(Vector input)
        {
            Vector output = null; // Объявление будущего выходного вектора

            for (int k = 0; k < layersN; k++)
            {
                output = new Vector(weights[k].n); // Создание нового выходного вектора для каждого слоя

                for (int i = 0; i < weights[k].n; i++)
                {
                    double y = 0; // Неактивированный выход нейрона

                    for (int j = 0; j < weights[k].m; j++)
                        y += weights[k][i, j] * input[j];

                    // Выполняем активацию с помощью сигмоидальной функции
                    output[i] = 1 / (1 + Math.Exp(-y));

                    // Выполняем активацию с помощью гиперболического тангенса
                    // output[i] = Math.Tanh(y);

                    // Выполняем активацию с помощью ReLU
                    // output[i] = Math.Max(0, y);
                }
            }

            return output;
        }
    }
}
