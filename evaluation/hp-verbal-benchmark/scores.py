def calculate_all(results):
    """Calculate accuracy for each task"""
    accuracy_ord = [calculate_accuracy(batch) for batch in results["ord"]]
    accuracy_läs = [calculate_accuracy(batch) for batch in results["läs"]]
    accuracy_mek = [calculate_accuracy(batch) for batch in results["mek"]]
    """Calculate the average accuracy for each task."""
    average_accuracy_ord = calculate_accuracy(accuracy_ord)
    average_accuracy_läs = calculate_accuracy(accuracy_läs)
    average_accuracy_mek = calculate_accuracy(accuracy_mek)
    """Calculate the total accuracy for the entire HP-test and take the average over all test accuracies."""
    total_acc = 0
    for i in range(10):
        total_acc += calculate_accuracy(results["ord"][i] + results["läs"][i] + results["mek"][i])
    average_accuracy_total = total_acc / 10

    return {"ord": accuracy_ord, 
            "läs": accuracy_läs,
            "mek": accuracy_mek,
            "avg_ord": average_accuracy_ord, 
            "avg_läs": average_accuracy_läs, 
            "avg_mek": average_accuracy_mek, 
            "avg_total": average_accuracy_total}

def calculate_accuracy(scores):
    return sum(scores) / len(scores)

def judge_answer(example, output) -> bool:
    return example['answer'] in output