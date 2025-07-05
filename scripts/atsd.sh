#!/bin/bash
CMD="python -m src.train_GeniusRec --encoder_weights_path checkpoints/hstu_encoder.pth" 
while pgrep -f "$CMD" > /dev/null; do
    sleep 3  
done

echo "程序已结束，正在关机..."
/usr/bin/shutdown -h now