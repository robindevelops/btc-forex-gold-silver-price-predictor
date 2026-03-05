first_number  = int(input("Enter a number: "))
second_number = int(input("Enter a number: "))
arithmetic_operation = input("Enter an arithmetic operation: ")


if (arithmetic_operation == "+"):
    print(first_number + second_number)
elif (arithmetic_operation == "-"):
    print(first_number - second_number)
elif (arithmetic_operation == "*"):
    print(first_number * second_number)
elif (arithmetic_operation == "/"):
    print(first_number / second_number)
else:
    print("Invalid operation")



