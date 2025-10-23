# 測試 multiprocessing 是否可用

'''
Results of top command (macOS):
60642 coconut     17   0  391G 15488 R   2.4  0.1  0:00.47 /opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -c from multiprocessing.
60640 coconut     17   0  391G 15408 R   2.4  0.1  0:00.48 /opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -c from multiprocessing.
60641 coconut     17   0  391G 15280 R   2.4  0.1  0:00.48 /opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -c from multiprocessing.
60639 coconut     20   0  391G 15008 R   2.4  0.1  0:00.48 /opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -c from multiprocessing.
60636 coconut     32   0  391G 14688 S   0.0  0.1  0:00.00 /opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python baseline/process.py
'''

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
def worker(num):
    """子進程工作函數"""
    print(f"子進程 {num} 開始工作")
    # 模擬一些工作
    total = 0
    for i in range(100000000000000000):
        total += i
    print(f"子進程 {num} 完成工作，結果總和為 {total}")
if __name__ == "__main__":
    processes = []
    num_processes = 4  # 啟動 4 個子進程

    for i in range(num_processes):
        p = multiprocessing.Process(target=worker, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("所有子進程工作完成")