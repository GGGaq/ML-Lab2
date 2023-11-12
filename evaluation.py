from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report,cohen_kappa_score
import matplotlib.pyplot as plt

def ModelEvaluation(y_true, y_pred, ModelName = ''):
    print(ModelName + ' Evaluation:')

    # 准确率
    acc = accuracy_score(y_pred = y_pred, y_true = y_true)
    print('accuracy score: ' + str(acc))

    # Keppa
    keppa = cohen_kappa_score(y_true, y_pred)
    print('keppa score: ' + str(keppa))

    # 分类报告
    print(classification_report(y_true = y_true, y_pred = y_pred))

    # 混淆矩阵
    confusion = confusion_matrix(y_true = y_true, y_pred = y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix = confusion)
    disp.plot()
    plt.title(ModelName)
    plt.savefig('output/' + ModelName)
    #plt.show()
    #plt.close()

