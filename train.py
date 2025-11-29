def train_epoch(device, epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    size = len(train_loader.dataset)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if device == 'cuda':
          inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 500 == 0:
          loss, current = loss.item(), batch_idx * len(inputs)
          print('loss: %.7f [%d/%d]' % (loss, current, size))

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'\nTrain Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

    return train_loss, train_accuracy