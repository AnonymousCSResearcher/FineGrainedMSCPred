from streamlit import bootstrap

# run viewer
real_script = 'src/viewer.py'
bootstrap.run(real_script, f'src/viewer.py {real_script}', [], {})