import time
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

if __name__ == '__main__':

    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transforms with Random Rotation and Random Horizontal Flip for training data
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor()
    ])

    # Create the datasets and dataloaders with updated batch size
    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)  ##### Increased batch size to 16
    testloader = DataLoader(testset, batch_size=1, shuffle=False)    ##### Increased batch size for test

    # Create the model and optimizer with updated learning rate
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.000075)  ##### Increased learning rate to 0.001

    # Define the loss functions
    criterion_ce = torch.nn.CrossEntropyLoss()  # For CrossEntropyLoss
    criterion_nll = torch.nn.NLLLoss()          # For NLLLoss with log_softmax

    # Loss lists for comparison
    results = {"CrossEntropyLoss": {"train_losses": [], "test_losses": [], "test_accuracies": []}, 
               "NLLLoss": {"train_losses": [], "test_losses": [], "test_accuracies": []}}

    # Function to run an epoch with a specified loss criterion
    def train_and_evaluate(criterion, loss_name):
        best_train_loss = 1e9

        for epoch in range(5):  # Using fewer epochs for comparison purposes

            # Start timer
            t = time.time_ns()

            # Train the model
            model.train()
            train_loss = 0

            # Batch Loop
            for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
                # Move data to the device
                images = images.reshape(-1, 3, 64, 64).to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)

                # Apply log_softmax if using NLLLoss
                if loss_name == "NLLLoss":
                    outputs = F.log_softmax(outputs, dim=1)

                # Compute the loss
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Accumulate the loss
                train_loss += loss.item()

            # Test the model
            model.eval()
            test_loss = 0
            correct = 0
            total = 0

            # Batch Loop for testing
            for images, labels in tqdm(testloader, total=len(testloader), leave=False):
                # Move data to the device
                images = images.reshape(-1, 3, 64, 64).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Apply log_softmax if using NLLLoss
                if loss_name == "NLLLoss":
                    outputs = F.log_softmax(outputs, dim=1)

                # Compute the loss
                loss = criterion(outputs, labels)

                # Accumulate the loss
                test_loss += loss.item()

                # Get the predicted class from the maximum value in the output list of class scores
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)

                # Accumulate the number of correct classifications
                correct += (predicted == labels).sum().item()

            # Calculate accuracy and save results
            test_accuracy = correct / total
            results[loss_name]["train_losses"].append(train_loss / len(trainloader))
            results[loss_name]["test_losses"].append(test_loss / len(testloader))
            results[loss_name]["test_accuracies"].append(test_accuracy)

            # Print epoch statistics
            print(f'{loss_name} - Epoch: {epoch}, Train Loss: {train_loss / len(trainloader):.4f}, '
                  f'Test Loss: {test_loss / len(testloader):.4f}, Test Accuracy: {test_accuracy:.4f}, '
                  f'Time: {(time.time_ns() - t) / 1e9:.2f}s')

            # Save the model
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save(model.state_dict(), f'best_model_{loss_name}.pth')

        # Save final model
        torch.save(model.state_dict(), f'final_model_{loss_name}.pth')

    # Train and evaluate with CrossEntropyLoss
    train_and_evaluate(criterion_ce, "CrossEntropyLoss")

    # Reinitialize the model and optimizer for NLLLoss
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  ##### Keep same learning rate for fair comparison

    # Train and evaluate with NLLLoss
    train_and_evaluate(criterion_nll, "NLLLoss")

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(results["CrossEntropyLoss"]["train_losses"], label='Train Loss - CrossEntropy')
    plt.plot(results["CrossEntropyLoss"]["test_losses"], label='Test Loss - CrossEntropy')
    plt.plot(results["NLLLoss"]["train_losses"], label='Train Loss - NLLLoss')
    plt.plot(results["NLLLoss"]["test_losses"], label='Test Loss - NLLLoss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('task8.png')
    plt.show()

    # Print final comparison of accuracies
    print("Final Test Accuracies:")
    print(f"CrossEntropyLoss: {results['CrossEntropyLoss']['test_accuracies'][-1]:.4f}")
    print(f"NLLLoss: {results['NLLLoss']['test_accuracies'][-1]:.4f}")