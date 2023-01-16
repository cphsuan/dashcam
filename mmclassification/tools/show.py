from mmcls.apis import init_model, inference_model, show_result_pyplot

config_file = 'resnet50_8xb32_in1k.py'
checkpoint_file = 'resnet50_8xb32_in1k_20210831-ea4938fc.pth'
model = init_model(config_file, checkpoint_file, device='cpu')  # 或者 device='cuda:0'
result = inference_model(model, 'demo/demo.JPEG')
show_result_pyplot(model, 'demo/demo.JPEG', result)