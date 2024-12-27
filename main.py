import src.capturing as capturing
import src.inference as inference


def main():
    print("Automated Checking Attendance System")
    while True:
        print("Please select a mode:")
        print("1. Capturing Mode")
        print("2. Inference Mode")
        print("3. Exit")

        try:
            choice = int(input(" [-] Enter mode: "))

            if choice == 1:
                print("You selected Capturing Mode.")
                capturing.capturing_mode()

            elif choice == 2:
                print("You selected Inferencing Mode. Implement the code here.")
                inference.load_memory()
                inference.start_inference()

            elif choice == 3:
                print("Exiting the program. Goodbye!")
                break

            else:
                print("Invalid choice. Please select 1, 2, or 3.")

        except ValueError:
            print("Invalid input. Please enter a number (1/2/3).")


if __name__ == "__main__":
    main()
