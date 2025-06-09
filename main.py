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

# ustawienie urzadzenia - gpu jesli dostepne, w przeciwnym razie cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# pobranie zbioru danych heart disease z repozytorium uci
heart_disease = fetch_ucirepo(id=45) 
  
# dane jako dataframe pandas
X = heart_disease.data.features 
y = heart_disease.data.targets 
y.columns = ['target']

# polaczenie cech i etykiet w jeden dataframe
data = pd.concat([X, y], axis=1 )

print(data)

# funkcja do analizy wykresow pudelkowych dla cech numerycznych
def boxplot_analysis(data):
    numeric_features = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']

    eda_df = data.loc[:, numeric_features].copy()

    plt.figure(figsize=(16, 10))

    # tworzenie wykresow pudelkowych dla kazdej cechy
    for i in range(len(eda_df.columns)):
        plt.subplot(2, 4, i + 1)
        sns.boxplot(eda_df[eda_df.columns[i]])

    plt.show()

# funkcja do analizy macierzy korelacji
def heatmap_analysis(data):
    numeric_features = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']

    eda_df = data.loc[:, numeric_features].copy()

    # obliczenie macierzy korelacji
    corr = eda_df.corr()

    # wizualizacja macierzy korelacji jako mapa ciepla
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, vmin=-1.0, cmap='mako')
    plt.title("Correlation Heatmap")
    plt.show()

# funkcja do analizy rozkladu klas w zbiorze danych
def pie_chart_analysis(data):
    plt.figure(figsize=(8, 8))
    plt.pie(data['target'].value_counts(), labels=[0, 1, 2, 3, 4], autopct='%.1f%%', colors=['#44ce1b', '#bbdb44', '#f7e379', '#f2a134', '#e51f1f'])
    plt.title("Class Distribution")
    plt.show()

# zamiana problemu wieloklasowego na binarny - 0 pozostaje 0, reszta staje sie 1
data['target'] = data['target'].apply(lambda x: 0 if x == 0 else 1)

# funkcja do kodowania one-hot zmiennych kategorycznych
def onehot_encode(df, column_dict):
    df = df.copy()
    for column, prefix in column_dict.items():
        # tworzenie zmiennych dummy dla kazdej kolumny kategorycznej
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df

# funkcja do preprocessingu danych
def preprocess_inputs(df, scaler):
    df = df.copy()
    
    # kodowanie one-hot dla cech nominalnych
    nominal_features = ['cp', 'slope', 'thal']
    df = onehot_encode(df, dict(zip(nominal_features, ['CP', 'SL', 'TH'])))
    
    # podzial na cechy i etykiety
    y = df['target'].copy()
    X = df.drop('target', axis=1).copy()
    
    # standaryzacja cech
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y

# preprocessing danych
X, y = preprocess_inputs(data, StandardScaler())

# podzial na zbiory treningowy i testowy (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)

# uzupelnienie brakujacych wartosci srednia
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_filled = imputer.fit_transform(X_train)
X_test_filled = imputer.transform(X_test)

# konwersja do tensorow pytorch i przeniesienie na odpowiednie urzadzenie
X_train_tensor = torch.FloatTensor(X_train_filled).to(device)
y_train_tensor = torch.LongTensor(y_train.values).to(device)
X_test_tensor = torch.FloatTensor(X_test_filled).to(device)
y_test_tensor = torch.LongTensor(y_test.values).to(device)

# utworzenie dataloader do treningu w batchach
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# definicja sieci neuronowej
class HeartDiseaseNet(nn.Module):
    def __init__(self, input_size):
        super(HeartDiseaseNet, self).__init__()
        # warstwy ukryte z 100 neuronami kazdej
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        # warstwa wyjsciowa z 2 neuronami dla klasyfikacji binarnej
        self.fc3 = nn.Linear(100, 2)
        self.relu = nn.ReLU()
        # dropout do regularyzacji (20% neuronow wylaczanych)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # propagacja w przod przez siec
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# inicjalizacja modelu
input_size = X_train_tensor.shape[1]
model = HeartDiseaseNet(input_size).to(device)

# funkcja straty i optymalizator
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# funkcja treningu modelu
def train_model(model, train_loader, criterion, optimizer, epochs=800):
    model.train()  # ustawienie modelu w tryb treningu
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            # zerowanie gradientow
            optimizer.zero_grad()
            # propagacja w przod
            outputs = model(batch_X)
            # obliczenie straty
            loss = criterion(outputs, batch_y)
            # propagacja wsteczna
            loss.backward()
            # aktualizacja wag
            optimizer.step()
            epoch_loss += loss.item()
        
        # obliczenie sredniej straty dla epoki
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        # wyswietlenie straty co 100 epok
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    return losses

# trenowanie modelu
print("Training")
losses = train_model(model, train_loader, criterion, optimizer, epochs=250)

# funkcja ewaluacji modelu
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
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Neural Network")
    print(f"Accuracy:  {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall:    {rec:.2f}")
    print(f"F1 Score:  {f1:.2f}")

    # tworzenie i wyswietlenie macierzy pomylek
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues")
    plt.title("Neural Network - Confusion Matrix")
    plt.savefig("./plots/NN_confusion_matrix.png")
    plt.show()
    
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
    plt.show()

# ewaluacja modelu
accuracy = evaluate_pytorch_model(model, X_test_tensor, y_test_tensor)
plot_learning_curve_pytorch(losses)

print(f"Neural Network Accuracy: {accuracy * 100:.2f}%")