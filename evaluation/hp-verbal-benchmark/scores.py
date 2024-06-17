def calculate_all(results):
    """Calculate the average accuracy for each task."""
    average_accuracy_ord = calculate_accuracy([calculate_accuracy(batch) for batch in results["ord"]])
    average_accuracy_läs = calculate_accuracy([calculate_accuracy(batch) for batch in results["läs"]])
    average_accuracy_mek = calculate_accuracy([calculate_accuracy(batch) for batch in results["mek"]])
    """Calculate the total accuracy for the entire HP-test and take the average over all test accuracies."""
    total_acc = 0
    for i in range(10):
        total_acc += calculate_accuracy(results["ord"][i] + results["läs"][i] + results["mek"][i])
    average_accuracy_total = total_acc / 10
    return {"ord": average_accuracy_ord, "läs": average_accuracy_läs, "mek": average_accuracy_mek, "total": average_accuracy_total}

def calculate_accuracy(scores):
    return sum(scores) / len(scores)

def judge_answer(example, output) -> bool:
    return example['answer'] in output