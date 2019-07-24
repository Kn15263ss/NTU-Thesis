# NTU-Thesis
Master thesis research

###---SVM---###

svm.py主要使用超參數優化的方法在svm模型上
輸入：python main.py --Dataset_name="dataset name"
輸出：ray_results

svm_test_result.py 從best_parameter資料夾中的valid載入model,並且test後得到結果, 路徑由base_path來修改


###----NN 專用 ----###

main.py為使用超參數優化的方法來訓練NN
輸入：python main.py --Dataset_name="dataset name" --Network_name="model name"
輸出：ray_results

search_best_parameter.py為尋找ray_results資料夾裡各個優化方法裡面最佳的超參數(用accuracy去判斷)是為何
輸入：python search_best_parameter.py --file_path="超參數優化方法的路徑"
example: '/home/willy-huang/workspace/research/with_pretrain/gridesearch_results'
輸出：best_parameter資料夾

search_best_loss_parameter.py同上, 不過是用loss作為判斷標準

search_top5_parameter.py尋找前5個accuracy最好的超參數, 主要用來比較random search和hyperband.
輸入：python search_best_parameter.py --file_path="超參數優化方法的路徑"
輸出：top5_parameter

nn_test_result.py 從best_parameter資料夾中的valid載入model,並且test後得到結果, 路徑由base_path來修改

test_parameter.py 測試從best_parameter中, 測試在不同的優化方法中, 最好的那一個優化方法的那一組超參數, 並且把那組超參數除了改變weight decay以外, 其餘都相同, 然後重新訓練存成新的資料, 主要是要用來比較有沒有包含weight decay的差異性.
輸入：python test_parameter.py --Dataset_name="dataset name" --Network_name="model name" --lr="value" --momentum="value" --weight_decay="value" --factor="value"
輸出：test_nn資料夾

plot_compare_result.py 畫出再相同的超參數組, 有沒有包含weight decay的差別, 對應test_parameter.py

plot_acc_and_loss.py 畫出在best_parameter資料夾中, 各個超參數優化方法的accuracy和loss
默認路徑由file_path來修改

plot_bayesian_eachfile.py 從bayesian optimization方法的資料夾中, 抽樣畫出不同超參數組
默認路徑由file_path來修改
舉例：bayesian現在有32組, 那麼我們想要畫的順序是[2,8,10,16,22,30]

plot_bayesian_best_eachfile.py 從bayesian optimization方法的資料夾中, 畫出最佳的超參數是哪一組
默認路徑由file_path來修改
舉例：bayesian現在有32組, 那麼我們想要畫的順序是[2,8,10,16,22,30], 就會從前2組選出最佳的, 再來從前8組中選出最佳的, 依此類推直到30組

plot_eachtop_acc_loss.py 畫出在不同的超參數優化方法中, 每一組(指超參數優化方法的資料夾裡面)在不同的epoch中, 表現最好那一組.
也就是畫出在同個超參數優化方法下, 每個epoch所表現最佳的超參數那一組
默認路徑由file_path來修改

plot_top5_acc_and_loss.py 畫出top5_parameter中的各個top資料, 主要對應search_top5_parameter.py
默認路徑由file_path來修改

###----其他----###

NN_example.py 一個簡單的測試網路

run_main.sh 用來跑main.py的script, 主要吃test.txt
alldata.txt >>> 為test.txt的完整版

run_search.sh 用來跑search_best_parameter.py的script

run_search_loss.sh 同上

run_svm.sh 類似run_main.sh, 不過是跑在SVM上, 吃test_svm.txt

run_test_parameter.sh 跑test_parameter.py的script, 吃test_parameter.txt

bayesian_source_code資料夾 在這裡不是指從github上下載的source code, 是我們有對裡面的取樣分佈特殊處理的source code, 主要有改變的資料是bayesian_optimization.py, target_space.py, util.py

query_hours_epoch.sh 查詢在ray_results中, 各個優化方法的總訓練時間(分hours跟epoch), 這裡我們的資料夾是optimization_method

query_param_acc_loss.sh 查詢在best_parameter資料夾中, 最佳的accuracy跟loss數值是多少, 並且附加查詢此超參數是多少數值

query_top5.sh 主要從top5_parameter資料夾中, 用各個top資料夾裡的超參數組去匹配在同樣在Hyperband_results的超參數組, 並且複製他們到此top資料夾的目錄底下.
