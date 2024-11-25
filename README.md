# SPAGHETTI DEMO
A Flask server implementation of the SPAGHETTI model for 3D chair generation and manipulation.

## Dependencies 
- Python >= 3.8
- PyTorch >= 1.8
- VTK
- scikit-image
- Flask
- Flask-CORS
- trimesh
- numpy
- open3d
- Pillow (PIL)
- pyglet

### Environment Setup
You can set up your environment using the provided environment file:
```bash
conda env create -f environment_droplet.yml
conda activate new_env
```

Or install dependencies manually:
```bash
pip install flask flask-cors trimesh numpy open3d Pillow pyglet
```
### Running the Server
After installing the dependencies, start the Flask server by running:
```bash
python app.py
```
The server will start on http://localhost:5000 by default.
### API Endpoints

- /mix (POST): Mix multiple chair designs with specified part weights
- /random_chair (POST): Generate a random chair design
- /archive (POST): Save a chair design to the archive
- /gpu_status (GET): Check GPU utilization status
- /nums (GET): Get available chair indices
- /values (GET): Get values for available chairs

Notes

- Make sure you have a compatible GPU for optimal performance
- The server uses port 5000 by default
- Check GPU memory usage through the /gpu_status endpoint
