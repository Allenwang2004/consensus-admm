# 測試 multithread 功能

'''
Results of running the code
USER      PID   TT   %CPU STAT PRI     STIME     UTIME COMMAND
coconut 58605 s021    0.0 S    31T   0:00.01   0:00.02 /opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python baseline/thread.py
        58605        26.2 S    31T   0:00.02   0:08.23
        58605        19.7 R    31T   0:00.03   0:08.52
        58605        25.7 S    31T   0:00.02   0:08.36
        58605        28.3 S    31T   0:00.03   0:08.45
'''

import threading
def worker(num):
    """線程工作函數"""
    print(f"線程 {num} 開始工作")
    # 模擬一些工作
    total = 0
    for i in range(100000000000000000):
        total += i
    print(f"線程 {num} 完成工作，結果總和為 {total}")

if __name__ == "__main__":
    threads = []
    num_threads = 4  # 啟動 4 個線程

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("所有線程工作完成")