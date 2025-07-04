# 修复 random.sample 接受 list 而不是 set 的问题
import re

def fix_sample_code():
    with open('/root/autodl-tmp/genius_rec-main/src/unified_evaluation.py', 'r') as file:
        content = file.read()

    # 替换所有 random.sample(candidate_ids, ...) 为 random.sample(list(candidate_ids), ...)
    modified_content = re.sub(
        r'random\.sample\(candidate_ids,', 
        r'random.sample(list(candidate_ids),', 
        content
    )

    with open('/root/autodl-tmp/genius_rec-main/src/unified_evaluation.py', 'w') as file:
        file.write(modified_content)

    print("✅ 修复完成：所有 random.sample 调用现在都接收 list 而不是 set")

fix_sample_code()