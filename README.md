# Uso de Unitree 4D LiDAR L2 para mineria.

## Tabla de contenido

- [Información general](#información-general)
- [Tecnologías utilizadas](#tecnologías-utilizadas)
- [Instalación del entorno desde cero](#instalación-del-entorno-desde-cero)

---

## Información general

## Tecnologías utilizadas

- Open3d
- Python 3
- Terminal (bash)
- Sistema operativo Linux

---

## Instalación del entorno desde cero

### 1. Instalar Python y pip3

```bash
sudo apt update
sudo apt install -y python3 python3-pip
```

Verificar instalación:

```bash
python3 --version
pip3 --version
```

### 2. Instalar cmake, open3d y build-essentials

```bash
sudo apt install cmake
sudo apt install build-essential
pip install open3d
```
---

## Compilacion de archivos cpp

```bash
mkdir build
cd build
cmake ..
make
```