import json
import os
from tqdm import tqdm

# 配置路径
data_dir = './spotify_data/data'
output_file = './spotify_data/spotify_trajectories.txt'
mapping_file = './spotify_data/uri_to_id.json'

uri_to_id = {}
curr_id = 0

print("开始处理 Spotify 百万歌单...")

with open(output_file, 'w') as f:
    # 遍历 1000 个 slice 文件
    for filename in tqdm(sorted(os.listdir(data_dir))):
        if filename.endswith('.json'):
            with open(os.path.join(data_dir, filename), 'r') as j:
                data = json.load(j)
                for playlist in data['playlists']:
                    # 过滤逻辑：只保留长度在 10-100 之间的优质"专家轨迹"
                    if 10 <= len(playlist['tracks']) <= 100:
                        track_ids = []
                        for track in playlist['tracks']:
                            uri = track['track_uri']
                            # 建立 URI 到整数 ID 的映射 (Action Tokenization)
                            if uri not in uri_to_id:
                                uri_to_id[uri] = curr_id
                                curr_id += 1
                            track_ids.append(str(uri_to_id[uri]))
                        
                        # 写入文件，每行一条轨迹，空格分隔
                        f.write(" ".join(track_ids) + "\n")

# 保存映射表，这相当于我们的 Action Vocabulary
with open(mapping_file, 'w') as m:
    json.dump(uri_to_id, m)

print(f"预处理完成！总计独立歌曲数量 (Action Space): {len(uri_to_id)}")
print(f"训练轨迹已保存至: {output_file}")
