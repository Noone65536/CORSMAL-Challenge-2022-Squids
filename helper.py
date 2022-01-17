import torch
import torch.nn as nn
from loss import computeScoreType1
import numpy as np

def train_audio(model, train_loader, optimizer, device, criterion = nn.CrossEntropyLoss()):
  model.train()
  loss_train = 0.0
  correct_train = 0.0
  num_train = len(train_loader)
  for batch_idx, (audio, target) in enumerate(train_loader):
    audio = audio.to(device)
    target = target.to(device)

    optimizer.zero_grad()
    outputs = model.forward(audio)

    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

    loss_train += loss.item() / num_train
    _, preds=torch.max(outputs,1)
    correct_train+=torch.sum(preds==target).item()
  
  return loss_train, correct_train


def evaluate_audio(model, testloader, device, criterion = nn.CrossEntropyLoss()):
  model.eval()
  loss_test = 0
  correct_test=0
  num_val = len(testloader)
  with torch.no_grad():
    for batch_idx, (audio, target) in enumerate(testloader):
      audio = audio.to(device)
      target = target.to(device)
      outputs = model.forward(audio)
      loss = criterion(outputs, target)
      loss_test += loss.item() / num_val
      _, preds=torch.max(outputs,1)
      correct_test+=torch.sum(preds==target).item()
  
  return loss_test, correct_test


def train_lstm(model, train_loader, optimizer, device, criterion = nn.CrossEntropyLoss()):
  model.train()
  loss_train = 0.0
  correct_train = 0.0
  num_train = len(train_loader)
  for batch_idx, (audio, target) in enumerate(train_loader):
    audio = audio.to(device)
    target = target.to(device)

    optimizer.zero_grad()
    outputs = model.forward(audio)

    loss = criterion(outputs, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    loss_train += loss.item() / num_train
    _, preds=torch.max(outputs,1)
    correct_train+=torch.sum(preds==target).item()
  
  return loss_train, correct_train


def train_image(model, train_loader, optimizer, device, criterion = nn.L1Loss()):
  model.train()
  loss_train = 0.0
  correct_train = 0.0
  num_train = len(train_loader)
  for batch_idx, (audio, target) in enumerate(train_loader):
    audio = audio.to(device)
    target = target.to(device)

    optimizer.zero_grad()
    outputs = model.forward(audio)
    loss = criterion(outputs, target)
    loss.backward()
    #nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    loss_train += loss.item() * audio.shape[0]
    correct_train += computeScoreType1(target, outputs) * audio.shape[0]

    
    
  
  return loss_train, correct_train


def evaluate_image(model, testloader, device, criterion = nn.L1Loss()):
  model.eval()
  loss_test = 0
  correct_test=0
  num_val = len(testloader)
  with torch.no_grad():
    for batch_idx, (audio, target) in enumerate(testloader):
      audio = audio.to(device)
      target = target.to(device)
      outputs = model.forward(audio)
      loss = criterion(outputs, target)
      loss_test += loss.item() * audio.shape[0]
      correct_test += computeScoreType1(target, outputs) * audio.shape[0]


  
  return loss_test, correct_test


def evaluate_var(model, testloader, device, criterion = nn.L1Loss()):
  model.eval()
  loss_test = 0
  correct_test=0
  num_val = len(testloader)
  results = {}
  variance = 0
  with torch.no_grad():
    for batch_idx, (audio, target) in enumerate(testloader):
      audio = audio.to(device)
      target = target.to(device)
      outputs = model.forward(audio)
      loss = criterion(outputs, target)
      loss_test += loss.item() * audio.shape[0]
      correct_test += computeScoreType1(target, outputs) * audio.shape[0]

      for i, c in enumerate(target):
        c = c.item()
        if c not in results.keys():
          results[c] = [outputs[i].item()]
        else:
          results[c].append(outputs[i].item())

    for k in results.keys():
      variance += np.mean(abs(np.array(results[c]) - k)**2).item()

    
    variance /= len(results.keys())
  
  return loss_test, correct_test, variance