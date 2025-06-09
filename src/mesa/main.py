from PolishEconomyModel import PolishEconomyModel

if __name__ == "__main__":
    print("Starting Enhanced RL Economy Simulation...")
    print("This may take a moment to complete...")

    # Initialize model
    model = PolishEconomyModel(1000,50)

    # Run simulation
    steps = 60
    for i in range(steps):
        model.step()
        if (i + 1) % 30 == 0:
            print(f"Step {i + 1}/{steps} completed")

    print("Simulation completed!")
    print("\n" + "="*50)


    print("\nGenerating comprehensive visualizations...")
    model.visualize_polish_results()

    print("Analysis complete! Check the generated plots above.")