def main():
    import os
    pre=set()
    if not os.path.exists("pre.txt"):
        print("build failed")
        return
    with open("pre.txt") as f:
        l=f.read().splitlines()
    for i in l:
        parts = i.strip().split()
        if len(parts) <= 1:
            continue
        if 'pass' in parts[-1].lower():
            pre.add(' '.join(parts[:-1]))
    if len(pre) == 0:
        print("build failed")
        return
    else:
        print("build success")

    post=set()
    if not os.path.exists("post.txt"):
        print("release failed")
        return
    with open("post.txt") as f:
        l=f.read().splitlines()
    for i in l:
        parts = i.strip().split()
        if len(parts) <= 1:
            continue
        if 'pass' in parts[-1].lower():
            post.add(' '.join(parts[:-1]))
    if len(post-pre) > 0:
        print("release success")
        print(post-pre)
    else:
        print("release failed")
    return

main()