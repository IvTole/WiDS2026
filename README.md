# WiDS2026

## Acerca de la Competencia

Este repositorio contiene el trabajo para la competencia **WiDS (Women in Data Science) Worldwide Global Datathon 2026**. La competencia se enfoca en resolver problemas del mundo real utilizando ciencia de datos, promoviendo la participación de mujeres en el campo de la ciencia de datos.

**Competencia:** [WiDS Worldwide Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/overview)

## Descarga de Datos

### Prerrequisitos
- Tener instalado Kaggle CLI: `pip install kaggle`
- Configurar las credenciales de Kaggle (archivo `kaggle.json` en `~/.kaggle/`)

### Procedimiento Estándar

1. **Crear la carpeta de datos:**
   ```bash
   mkdir -p data
   ```

2. **Descargar los datos de la competencia:**
   ```bash
   kaggle competitions download -c WiDSWorldWide_GlobalDathon26 -p data/
   ```

3. **Extraer los archivos (si es necesario):**
   ```bash
   cd data
   unzip "*.zip"
   ```

Los datos se descargarán en la carpeta `/data` del proyecto.