import Testing


def main():
    results_Pen = []
    for i in range(5):
        results_Pen.append(Testing.testPenData()[1])
    print("=======================================================")
    print("|  Pen Results:")
    print("|          MAX:" + str(max(results_Pen)))
    print("|      Average:" + str(Testing.average(results_Pen)))
    print("|      Std Dev:" + str(Testing.stDeviation(results_Pen)))
    print("=======================================================")
    results_Car = []
    for i in range(5):
        results_Car.append(Testing.testCarData()[1])
    print("=======================================================")
    print("|  Car Results:")
    print("|         MAX:" + str(max(results_Car)))
    print("|     Average:" + str(Testing.average(results_Car)))
    print("|     Std Dev:" + str(Testing.stDeviation(results_Car)))
    print("=======================================================")

main()