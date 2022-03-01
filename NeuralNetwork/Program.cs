using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            // Массив входных обучающих векторов
            Vector[] X = {
                new Vector(0, 0),
                new Vector(0, 1),
                new Vector(1, 0),
                new Vector(1, 1)
            };

            // Массив выходных обучающих векторов
            Vector[] Y = {
                new Vector(0.0), // 0 ^ 0 = 0
                new Vector(1.0), // 0 ^ 1 = 1
                new Vector(1.0), // 1 ^ 0 = 1
                new Vector(0.0) // 1 ^ 1 = 0
            };

            Network network = new Network(new int[]{2, 3, 1}); // Создаём сеть с двумя входами, тремя нейронами в скрытом слое и одним выходом

            network.Train(X, Y, 0.5, 1e-7, 100000); // Запускаем обучение сети 

            // Просмотр результатов после обучения с помощью выполнения прямого прохода для всех элементов
            for (int i = 0; i < 4; i++)
            {
                Vector output = network.Forward(X[i]);
                Console.WriteLine($"X: {X[i][0]} {X[i][1]}, Y: {Y[i][0]}, output: {output[0]}");
            }

            Console.WriteLine("Конец программы. Для выхода из неё нажмите любую клавишу на клавиатуре.");
            Console.ReadKey();
        }
    }
}
