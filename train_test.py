from data import *
import json

seeds = list(range(3))
epochs_add = [100, 400, 500, 1000, 2000, 6000]
epochs_total = [100, 500, 1000, 2000, 4000, 10000]
ratios = [0.25,0.5,1,2,4]
margins = [.1,.5,1,1.5]
for ratio in ratios:
    data = {}
    for margin in margins:
        data[str(margin)] = {}
        for seed in seeds:
            data[str(margin)][seed] = {}
            X, y = data_gen((0,1,ratio), num_negative=100, num_positive=100, margin=margin,seed=seed)
            y_01 = (y+1)/2
            save_dataset(X,y,f"dataset/data-r{ratio}-m{margin}-s{seed}.npy")
            svm_model, svm_trained_plane = train_svm_get_hyperplane(X, y)
            model_logistic = None
            for i, epoch in enumerate(epochs_add):
                model_logistic, trained_plane_logistic = train_logistic_regression_gd(X,y_01,epoch, 0.01, model_logistic)
                plot_line_from_coeffs([svm_trained_plane, trained_plane_logistic], X,y,f'sresult/r{ratio}-m{margin}-s{seed}-e{epochs_total[i]}.png')
                data[str(margin)][seed][epochs_total[i]] = str(trained_plane_logistic)
        print(f"r{ratio}-m{margin} completed")
    with open(f'sgd_r{ratio}result.json','w') as fp:
        json.dump(data,fp)
