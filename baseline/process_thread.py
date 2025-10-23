# 測試在 multiprocessing 內再啟用 multithreading 是否可用
import multiprocessing
import threading

'''
Results of top command (macOS) when running this code:
62684 coconut     24   0  391G 15808 R   2.4  0.1  0:00.69 /opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -c from multiprocessing.
62686 coconut     24   0  391G 15552 R   2.4  0.1  0:00.69 /opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -c from multiprocessing.
62683 coconut     24   0  391G 15184 R   2.4  0.1  0:00.69 /opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -c from multiprocessing.
62685 coconut     24   0  391G 15120 R   2.4  0.1  0:00.69 /opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -c from multiprocessing.
62681 coconut     32   0  391G 14064 S   0.0  0.1  0:00.00 /opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python baseline/process_thread.
                                                                                                                                   ok  base py  09:43:43
USER      PID   TT   %CPU STAT PRI     STIME     UTIME COMMAND
coconut 62686 s021    0.0 S    31T   0:00.01   0:00.02 /opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -c from multiprocessing.spa
        62686        25.6 R    31T   0:00.04   0:03.95
        62686        24.5 S    31T   0:00.04   0:03.98
        62686        24.4 S    31T   0:00.04   0:03.94
        62686        25.3 S    31T   0:00.04   0:03.96
'''

def thread_worker(num):
    """線程工作函數"""
    print(f"    線程 {num} 開始工作")
    # 模擬一些工作
    total = 0
    for i in range(100000000000000000):
        total += i
    print(f"    線程 {num} 完成工作，結果總和為 {total}")

def process_worker(num):
    """子進程工作函數"""
    print(f"子進程 {num} 開始工作")
    
    # 在子進程內啟動多個線程
    threads = []
    num_threads = 4  # 每個子進程啟動 4 個線程

    for i in range(num_threads):
        t = threading.Thread(target=thread_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"子進程 {num} 所有線程工作完成")

if __name__ == "__main__":
    processes = []
    num_processes = 4  # 啟動 4 個子進程

    for i in range(num_processes):
        p = multiprocessing.Process(target=process_worker, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("所有子進程工作完成")