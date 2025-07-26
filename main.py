import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from ucimlrepo import fetch_ucirepo 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

heart_disease = fetch_ucirepo(id=45) 
  

X = heart_disease.data.features 
y = heart_disease.data.targets 
y.columns = ['target']

# polaczenie cech i etykiet w jeden dataframe
data = pd.concat([X, y], axis=1 )

print(data)

#analiza wykresow pudelkowych 
def boxplot_analysis(data):
    numeric_features = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']

    eda_df = data.loc[:, numeric_features].copy()

    plt.figure(figsize=(16, 10))

    # tworzenie wykresow pudelkowych dla kazdej cechy
    for i in range(len(eda_df.columns)):
        plt.subplot(2, 4, i + 1)
        sns.boxplot(eda_df[eda_df.columns[i]])

    # plt.show()

def heatmap_analysis(data):
    numeric_features = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']

    eda_df = data.loc[:, numeric_features].copy()

    #obliczenie macierzy korelacji
    corr = eda_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, vmin=-1.0, cmap='mako')
    plt.title("Correlation Heatmap")
    # plt.show()

#rozklad klas
def pie_chart_analysis(data):
    plt.figure(figsize=(8, 8))
    plt.pie(data['target'].value_counts(), labels=[0, 1, 2, 3, 4], autopct='%.1f%%', colors=['#44ce1b', '#bbdb44', '#f7e379', '#f2a134', '#e51f1f'])
    plt.title("Class Distribution")
    # plt.show()

#zmiana z klas 0-4 na 0-1
# data['target'] = data['target'].apply(lambda x: 0 if x == 0 else 1)


def onehot_encode(df, column_dict):
    df = df.copy()
    for column, prefix in column_dict.items():
        # tworzenie zmiennych dummy dla kazdej kolumny kategorycznej
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df

#preprocessing wejsciowych
def preprocess_inputs(df, scaler):
    df = df.copy()
    
    # kodowanie one-hot
    nominal_features = ['cp', 'slope', 'thal']
    df = onehot_encode(df, dict(zip(nominal_features, ['CP', 'SL', 'TH'])))
    
    y = df['target'].copy()
    X = df.drop('target', axis=1).copy()
    
    # standaryzacja cech
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y

# preprocessing danych
X, y = preprocess_inputs(data, StandardScaler())

#podzial na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)

# uzupelnienie brakujacych wartosci srednia
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_filled = imputer.fit_transform(X_train)
X_test_filled = imputer.transform(X_test)

#konwersja do tensorow pytorch i przeniesienie na odpowiednie urzadzenie
X_train_tensor = torch.FloatTensor(X_train_filled).to(device)
y_train_tensor = torch.LongTensor(y_train.values).to(device)
X_test_tensor = torch.FloatTensor(X_test_filled).to(device)
y_test_tensor = torch.LongTensor(y_test.values).to(device)

# utworzenie dataloader do treningu w batchach
train_dataset = TensorDataset(X_train_tensor, y_train_tensor) #pytorch potrzebuje dataset, czyli łączenie cech i targetu
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # podczas jednej iteracji siec dsotanie 32 próbki


class HeartDiseaseNet(nn.Module):
    def __init__(self, input_size):
        super(HeartDiseaseNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        # warstwy ukryte
        # self.fc2 = nn.Linear(30, 30)
        # warstwa wyjsciowa z 2 neuronami dla klasyfikacji binarnej
        self.fc2 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        # dropout do regularyzacji (20% neuronow wyłąćzanych)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # propagacja w przod przez siec
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.fc3(x)
        return x

# inicjalizacja modelu
input_size = X_train_tensor.shape[1] # rozmiar wejscia - liczba cech
model = HeartDiseaseNet(input_size).to(device)

# funkcja straty i optymalizator
criterion = nn.CrossEntropyLoss()  #jak bardzo przewidywanie jest zblizone do rzeczywistej wartosci
optimizer = optim.Adam(model.parameters(), lr=0.001) #adaptacyjne wspolczynniki uczenia

def train_model(model, train_loader, criterion, optimizer, X_test, y_test, epochs=800):
    model.train()
    losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0
        
        #uczenie
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            


            _, predicted = torch.max(outputs.data, 1) #przewidywanie klasy z najwyzszym prawdopodobienstwem
            total_train += batch_y.size(0) # liczba probek w batchu
            correct_train += (predicted == batch_y).sum().item() # liczba poprawnych przewidywan
        
        avg_loss = epoch_loss / len(train_loader) # srednia strata dla epoki
        train_acc = correct_train / total_train # dokladnosc na zbiorze treningowym
        losses.append(avg_loss) 
        train_accuracies.append(train_acc) 
        
        
        model.eval() #wlaczanie trybu testowania
        with torch.no_grad(): # wylaczenie gradientow dla ewaluacji
            test_outputs = model(X_test)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_acc = (test_predicted == y_test).sum().item() / y_test.size(0)
            test_accuracies.append(test_acc)
        model.train()
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    return losses, train_accuracies, test_accuracies


def evaluate_pytorch_model(model, X_test, y_test):
    model.eval()  # ustawienie modelu w tryb ewaluacji
    with torch.no_grad():  # wylaczenie gradientow dla ewaluacji
        outputs = model(X_test)
        # wybor klasy z najwyzsza prawdopodobienstwem
        _, predicted = torch.max(outputs.data, 1)
        
    # konwersja do numpy dla metryk sklearn
    y_pred = predicted.cpu().numpy()
    y_true = y_test.cpu().numpy()
    
    # obliczenie metryk
    acc = accuracy_score(y_true, y_pred)
    
    #ponizsze statystyki sa dostepne tylko dla danych binarnych
    # prec = precision_score(y_true, y_pred)
    # rec = recall_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred)

    print(f"Neural Network")
    print(f"Accuracy:  {acc:.2f}")
    # print(f"Precision: {prec:.2f}")
    # print(f"Recall:    {rec:.2f}")
    # print(f"F1 Score:  {f1:.2f}")

    # tworzenie i wyswietlenie macierzy pomylek
    # cm = confusion_matrix(y_true, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    # disp.plot(cmap="Blues")
    # plt.title("Neural Network - Confusion Matrix")
    # plt.savefig("./plots/NN_confusion_matrix.png")
    # plt.show()
    
    return acc

# funkcja do wykresu krzywej uczenia
def plot_learning_curve_pytorch(losses):
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("Neural Network Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("./plots/learning_curve.png")
    # plt.show()

# funkcja do wykresu krzywej uczenia z loss i accuracy
def plot_learning_curves(losses, train_accuracies, test_accuracies):
    # loss
    plt.figure(figsize=(8, 6))
    plt.plot(losses, 'g-', linewidth=2)
    plt.title("Training Loss", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./plots/training_loss.png", dpi=300, bbox_inches='tight')
    # plt.show()
  
  # accuracy
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, 'b-', label='Training', linewidth=2)
    plt.plot(epochs, test_accuracies, 'r-', label='Test', linewidth=2)
    plt.title("Model Accuracy", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("./plots/model_accuracy.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    #roznica accuracy treningu i testu
    plt.figure(figsize=(8, 6))
    acc_diff = [train - test for train, test in zip(train_accuracies, test_accuracies)]
    plt.plot(epochs, acc_diff, 'purple', linewidth=2)
    plt.title("Training - Test Accuracy Difference", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Difference")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("./plots/accuracy_difference.png", dpi=300, bbox_inches='tight')
    # plt.show()

print("Training")
losses, train_accuracies, test_accuracies = train_model(
    model, train_loader, criterion, optimizer, X_test_tensor, y_test_tensor, epochs=51
)

# ewaluacja modelu i wykresy
accuracy = evaluate_pytorch_model(model, X_test_tensor, y_test_tensor)

# plot_learning_curves(losses, train_accuracies, test_accuracies)

print(f"Neural Network Final Training Accuracy: {train_accuracies[-1] * 100:.2f}%")
print(f"Neural Network Final Test Accuracy: {test_accuracies[-1] * 100:.2f}%")
print(f"Neural Network Final Accuracy: {accuracy * 100:.2f}%")

def weight_analysis(model):
    # analiza wag modelu
    weights = model.fc1.weight.data.cpu().numpy()
    plt.figure(figsize=(12, 6))
    sns.heatmap(weights, annot=True, cmap='coolwarm', cbar=True)
    plt.title("Weights of the First Layer")
    plt.xlabel("Features")
    plt.ylabel("Neurons")
    plt.tight_layout()
    plt.savefig("./plots/weights_analysis.png", dpi=300, bbox_inches='tight')
    # plt.show()

weight_analysis(model)
