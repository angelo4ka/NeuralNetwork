using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>
    /// Матрица (каждый слой содержит матрицу весовых коэффициентов)
    /// </summary>
    class Matrix
    {
        double[][] v; // Значения матрицы
        public int n, m; // Количество строк и столбцов

        /// <summary>
        /// Создание матрицы заданного размера и заполнение случайными числами из интервала (-0.5, 0.5)
        /// </summary>
        /// <param name="n">Количество строк</param>
        /// <param name="m">Количество столбцов</param>
        /// <param name="random">Генератор случайных чисел</param>
        public Matrix(int n, int m, Random random)
        {
            this.n = n; // Копируем количество строк
            this.m = m; // Копируем количество столбцов

            v = new double[n][]; // Создаём матрицу

            for (int i = 0; i < n; i++)
            {
                v[i] = new double[m];

                // Заполняем матрицу случайными числами
                for (int j = 0; j < m; j++)
                    v[i][j] = random.NextDouble() - 0.5;
            }
        }

        /// <summary>
        /// Обращение к значению матрицы по индексам
        /// </summary>
        /// <param name="i">Индекс строки</param>
        /// <param name="j">Индекс столбца</param>
        /// <returns>Значение матрицы</returns>
        public double this[int i, int j]
        {
            get { return v[i][j]; } // Получение значения
            set { v[i][j] = value; } // Изменение значения
        }
    }
}
