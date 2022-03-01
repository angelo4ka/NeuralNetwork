using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>
    /// Вектор (входные, выходные)
    /// </summary>
    class Vector
    {
        public double[] v; // Значения вектора
        public int n; // Длина вектора

        /// <summary>
        /// Создание вектора из количества элементов (длины)
        /// </summary>
        /// <param name="n">Длина вектора</param>
        public Vector(int n)
        {
            this.n = n; // Копируем длину
            v = new double[n]; // Создаём массив
        }

        /// <summary>
        /// Создание вектора из перечисления вещественных чисел (значений)
        /// </summary>
        /// <param name="values">Массив вещественных чисел</param>
        public Vector(params double[] values)
        {
            n = values.Length; // Находим длину 
            v = new double[n]; // Создаём массив

            // Заполняем массив
            for (int i = 0; i < n; i++)
                v[i] = values[i];
        }

        /// <summary>
        /// Обращение к значению вектора по индексу
        /// </summary>
        /// <param name="i">Индекс</param>
        /// <returns>Значение вектора</returns>
        public double this[int i]
        {
            get { return v[i]; } // Получение значения
            set { v[i] = value; } // Изменение значения
        }
    }
}
