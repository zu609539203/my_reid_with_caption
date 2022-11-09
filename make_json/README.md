# make_json

对人物框生成一句文字描述，并以`json`的格式保存

> 若是该人物已存在真实描述，则直接使用真实描述；若是没有则采用`caption`生成

```bash
格式：
    {
        'file_name' : 'xxx.jpg',
        'captions' : 'The man xxx xx xxx'
    }
```

