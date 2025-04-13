

N = 256
M = 256
K = 256
BLOCK_N = 4
BLOCK_M = 4
BLOCK_K = 1

TG_X = 4
TG_Y = 4
TDIM_X = 16
TDIM_Y = 16


def load_fn(src_a, dst_a, log_src_a, log_dst_a, val_a,
            src_b, dst_b, log_src_b, log_dst_b, val_b,
            tx, ty, tigx, tigy, i):
    for j in range(tigy, BLOCK_N * BLOCK_K, TDIM_Y):
        dst_idx = tigx * BLOCK_N * BLOCK_K + j
        src_idx = (tx * BLOCK_N + j // BLOCK_K) * K + i + j % BLOCK_K
        logging = f"idx: {src_idx}=>{dst_idx} [{tx}, {ty}], [{tigx}, {tigy}], {i},{j}"
        dst_a[dst_idx] += 1
        src_a[src_idx] += 1
        val_a[dst_idx] = f"[{src_idx // K}, {src_idx % K}]"
        log_dst_a[dst_idx].append(logging)
        log_src_a[src_idx].append(logging)

    for j in range(tigx, BLOCK_N * BLOCK_K, TDIM_X):
        dst_idx = tigy * BLOCK_N * BLOCK_K + j
        src_idx = (ty * BLOCK_M + j // BLOCK_K) * K + i + j % BLOCK_K
        logging = f"idx: {src_idx}=>{dst_idx} [{tx}, {ty}], [{tigx}, {tigy}], {i},{j}"
        dst_b[dst_idx] += 1
        src_b[src_idx] += 1
        val_b[dst_idx] = f"[{src_idx // K}, {src_idx % K}]"
        log_dst_b[dst_idx].append(logging)
        log_src_b[src_idx].append(logging)



def mul_fn(mul_a, mul_b, res, val_a, val_b, val, tx, ty, tigx, tigy, i):
    for x in range(BLOCK_N):
        for y in range(BLOCK_M):
            res_ptr = (tx * BLOCK_N + x) * M + ty * BLOCK_M + y

            for k in range(BLOCK_K):
                ptr_a = (tigx * BLOCK_N + x) * BLOCK_K + k
                ptr_b = (tigy * BLOCK_N + y) * BLOCK_K + k
                mul_a[ptr_a] += 1
                res[res_ptr] += 1
                val[res_ptr] += f"+ {val_a[ptr_a]} * {val_b[ptr_b]}"


def check(data, num_col, val=1):
    cnt = 0
    for i in range(len(data)):
        # print(f"{i} [{i // num_col}, {i % num_col}]\t:\t{data[i]}\t{data[i] == val}")
        if data[i] != val:
            cnt += 1
        

    if cnt == 0:
        print("PASS")
    else:
        print("FAIL")


def load():
    dst_a = []
    src_a = []
    log_dst_a = []
    log_src_a = []
    val_a = []

    dst_b = []
    src_b = []
    log_dst_b = []
    log_src_b = []
    val_b = []

    mul_a = []
    mul_b = []
    val = []
    res = []

    
    for i in range(N):
        for j in range(K):
            src_a.append(0)
            log_src_a.append([])
    for i in range(TDIM_X * BLOCK_N):
        for j in range(BLOCK_K):
            dst_a.append(0)
            val_a.append("")
            log_dst_a.append([])
            mul_a.append(0)

    for i in range(M):
        for j in range(K):
            src_b.append(0)
            log_src_b.append([])
    for i in range(TDIM_X * BLOCK_N):
        for j in range(BLOCK_K):
            dst_b.append(0)
            val_b.append("")
            log_dst_b.append([])
            mul_b.append(0)
    
    for i in range(N):
        for j in range(M):
            res.append(0)
            val.append("")


    for i in range(0, K, BLOCK_K):
        for x in range(TG_X):
            for y in range(TG_Y):
                for tigx in range(TDIM_X):
                    for tigy in range(TDIM_Y):
                        tx = x * TDIM_X + tigx
                        ty = y * TDIM_Y + tigy
                        #print(f"Thread: [{tx}, {ty}] - [{tigx}, {tigy}]")
                        load_fn(src_a, dst_a, log_src_a, log_dst_a, val_a,
                                src_b, dst_b, log_src_b, log_dst_b, val_b,
                                tx, ty, tigx, tigy, i)
                for tigx in range(TDIM_X):
                    for tigy in range(TDIM_Y):
                        tx = x * TDIM_X + tigx
                        ty = y * TDIM_Y + tigy
                        mul_fn(mul_a, mul_b, res, val_a, val_b, val, tx, ty, tigx, tigy, i)

    check(dst_a, BLOCK_K, TG_X * TG_Y * K / BLOCK_K)
    check(src_a, K, TG_Y)
    check(dst_b, BLOCK_K, TG_X * TG_Y * K / BLOCK_K)
    check(src_b, K, TG_Y)
    check(mul_a, BLOCK_K, M * N * M / (TDIM_X * BLOCK_N * BLOCK_K))
    check(res, M, K)
    print(val[0])
    print(val[30])

load()