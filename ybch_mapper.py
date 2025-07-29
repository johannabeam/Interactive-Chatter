import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configure the page
st.set_page_config(
    page_title="Yellow-breasted Chat Genoscape",
    page_icon="🐦",
    layout="wide"
)

st.title("🐦 Yellow-breasted Chat Genoscape")
st.markdown("Interactive map exploring genetic data and inversion genotypes across sampling locations")

# Your GitHub data URL - update this with your actual URL
GITHUB_DATA_URL = "https://raw.githubusercontent.com/johannabeam/Interactive-Chatter/refs/heads/main/YBCHall_samples_metadata.csv"

# Option to override the default URL
st.sidebar.header("📡 Data Source")
github_url = st.sidebar.text_input(
    "GitHub Raw CSV URL:",
    value=GITHUB_DATA_URL,
    help="Your GitHub raw CSV URL"
)

@st.cache_data
def load_github_data(url):
    """Load data from GitHub with caching"""
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading data from GitHub: {e}")
        return None

@st.cache_data
def load_grid_file_simple(grid_file):
    """Load ASCII grid file without rasterio"""
    try:
        # Read the file content
        content = grid_file.getvalue().decode('utf-8')
        lines = content.strip().split('\n')
        
        # Try to parse as ASCII grid format
        header = {}
        data_start = 0
        
        # Look for header information
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'nodata']):
                parts = line.split()
                if len(parts) >= 2:
                    header[parts[0].lower()] = float(parts[1]) if '.' in parts[1] or 'e' in parts[1].lower() else int(parts[1])
                data_start = i + 1
            else:
                break
        
        # Read the grid data
        grid_data = []
        for line in lines[data_start:]:
            if line.strip():
                row = [float(x) for x in line.split()]
                grid_data.append(row)
        
        if not grid_data:
            st.error("No grid data found in file")
            return None, None, None, None
        
        grid_array = np.array(grid_data)
        
        # Create coordinate arrays
        if 'ncols' in header and 'nrows' in header:
            ncols = int(header['ncols'])
            nrows = int(header['nrows'])
            
            if 'xllcorner' in header and 'yllcorner' in header and 'cellsize' in header:
                xll = header['xllcorner']
                yll = header['yllcorner']
                cellsize = header['cellsize']
                
                lon = np.linspace(xll, xll + ncols * cellsize, ncols)
                lat = np.linspace(yll, yll + nrows * cellsize, nrows)
                
                # Handle nodata values
                if 'nodata_value' in header:
                    nodata = header['nodata_value']
                    grid_array[grid_array == nodata] = np.nan
                
                return grid_array, lon, lat, (xll, yll, xll + ncols * cellsize, yll + nrows * cellsize)
        
        # If no proper header, try to use the data as-is
        nrows, ncols = grid_array.shape
        lon = np.linspace(0, ncols, ncols)
        lat = np.linspace(0, nrows, nrows)
        
        return grid_array, lon, lat, (0, 0, ncols, nrows)
        
    except Exception as e:
        st.error(f"Error reading grid file: {e}")
        return None, None, None, None

# Load data from GitHub
if github_url:
    df = load_github_data(github_url)
else:
    df = None

# Process data if available
if df is not None:
    # Clean and prepare data
    try:
        # Ensure we have the required columns
        required_cols = ['BGP_ID', 'Lat', 'Long', 'inv_k3', 'State', 'CityTown']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.write("Available columns:", list(df.columns))
        else:
            # Clean coordinate data
            df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
            df['Long'] = pd.to_numeric(df['Long'], errors='coerce')
            df_clean = df.dropna(subset=['Lat', 'Long', 'BGP_ID'])
            
            st.success(f"✅ Successfully loaded {len(df_clean)} samples with coordinates")
            
            # Show data preview
            with st.expander("📊 Data Preview"):
                st.dataframe(df_clean.head(10))
            
            # Sidebar controls
            st.sidebar.header("🎛️ Map Controls")
            
            # Grid overlay option
            st.sidebar.header("🌍 Grid Overlay")
            grid_overlay = st.sidebar.checkbox("Add grid overlay", value=False)
            
            grid_file = None
            if grid_overlay:
                grid_file = st.sidebar.file_uploader(
                    "Upload grid file (.grd, .txt, .asc)",
                    type=['grd', 'txt', 'asc'],
                    help="Upload ASCII grid file"
                )
                
                if grid_file:
                    grid_opacity = st.sidebar.slider(
                        "Grid opacity",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.6,
                        step=0.1
                    )
                    
                    grid_colorscale = st.sidebar.selectbox(
                        "Grid color scheme:",
                        ["Viridis", "Plasma", "RdYlBu", "RdBu", "Spectral", "Terrain"]
                    )
            
            # Color options
            color_options = {
                'Inversion Genotype (inv_k3)': 'inv_k3',
                'State': 'State', 
                'Population': 'pop1',
                'Climate': 'Climate',
                'Sex': 'Sex'
            }
            
            # Filter available color options based on what's in the data
            available_color_options = {k: v for k, v in color_options.items() if v in df_clean.columns}
            
            color_by = st.sidebar.selectbox(
                "Color points by:",
                list(available_color_options.keys()),
                index=0
            )
            color_column = available_color_options[color_by]
            
            # Size options
            size_options = {
                'Uniform': None,
                'Mean Depth': 'MEAN_DEPTH',
                'Coverage': 'coverage',
                'DNA Quantity': 'Final DNA Quant'
            }
            
            available_size_options = {k: v for k, v in size_options.items() if v is None or v in df_clean.columns}
            
            size_by = st.sidebar.selectbox(
                "Size points by:",
                list(available_size_options.keys()),
                index=0
            )
            size_column = available_size_options[size_by]
            
            # Map style
            map_style = st.sidebar.selectbox(
                "Map style:",
                ["open-street-map", "satellite-streets", "stamen-terrain", "carto-positron"]
            )
            
            # Load grid data if selected
            grid_data = None
            if grid_overlay and grid_file:
                with st.spinner("Loading grid data..."):
                    grid_values, grid_lon, grid_lat, grid_bounds = load_grid_file_simple(grid_file)
                    if grid_values is not None:
                        st.success("✅ Grid data loaded successfully")
                        grid_data = (grid_values, grid_lon, grid_lat, grid_bounds)
                        
                        # Show grid info
                        with st.expander("📊 Grid Data Info"):
                            st.write(f"Grid dimensions: {grid_values.shape}")
                            st.write(f"Value range: {np.nanmin(grid_values):.3f} to {np.nanmax(grid_values):.3f}")
                            st.write(f"Bounds: {grid_bounds}")
            
            # Create the interactive map
            st.header("🗺️ Interactive Genetic Data Map")
            
            # Create the map figure
            fig = go.Figure()
            
            # Add grid overlay first (so points appear on top)
            if grid_data:
                grid_values, grid_lon, grid_lat, grid_bounds = grid_data
                
                # Add grid as heatmap
                fig.add_trace(go.Heatmap(
                    x=grid_lon,
                    y=grid_lat, 
                    z=grid_values,
                    colorscale=grid_colorscale.lower(),
                    opacity=grid_opacity,
                    showscale=True,
                    colorbar=dict(
                        title="Grid Values",
                        x=1.02,
                        len=0.5
                    ),
                    hovertemplate='Lat: %{y:.4f}<br>Lon: %{x:.4f}<br>Value: %{z:.3f}<extra></extra>',
                    name="Grid Data"
                ))
            
            # Prepare hover data with your specified fields
            hover_data = []
            for _, row in df_clean.iterrows():
                hover_info = [
                    f"<b>{row['BGP_ID']}</b>",
                    f"Inversion: {row['inv_k3']}",
                    f"Location: {row['CityTown']}, {row['State']}",
                    f"Coordinates: {row['Lat']:.4f}, {row['Long']:.4f}"
                ]
                
                # Add additional info if available
                if 'Population' in row and pd.notna(row['Population']):
                    hover_info.append(f"Population: {row['Population']}")
                if 'Sex' in row and pd.notna(row['Sex']):
                    hover_info.append(f"Sex: {row['Sex']}")
                if 'Climate' in row and pd.notna(row['Climate']):
                    hover_info.append(f"Climate: {row['Climate']}")
                if 'MEAN_DEPTH' in row and pd.notna(row['MEAN_DEPTH']):
                    hover_info.append(f"Mean Depth: {row['MEAN_DEPTH']:.2f}")
                
                hover_data.append("<br>".join(hover_info))
            
            # Create the map
            fig = go.Figure()
            fig = go.Figure()
            
            # Handle coloring
            if color_column in df_clean.columns:
                if df_clean[color_column].dtype in ['object', 'category']:
                    # Categorical coloring
                    unique_vals = df_clean[color_column].unique()
                    # Use a nice color palette
                    colors = px.colors.qualitative.Set3
                    if len(unique_vals) > len(colors):
                        colors = colors * (len(unique_vals) // len(colors) + 1)
                    color_map = dict(zip(unique_vals, colors[:len(unique_vals)]))
                    point_colors = [color_map.get(val, 'gray') for val in df_clean[color_column]]
                    
                    # Add legend by creating separate traces for each category
                    for i, val in enumerate(unique_vals):
                        mask = df_clean[color_column] == val
                        if mask.sum() > 0:
                            subset_df = df_clean[mask]
                            subset_hover = [hover_data[j] for j, m in enumerate(mask) if m]
                            
                            # Handle sizing for this subset
                            if size_column and size_column in df_clean.columns:
                                sizes = subset_df[size_column]
                                sizes = ((sizes - df_clean[size_column].min()) / 
                                        (df_clean[size_column].max() - df_clean[size_column].min()) * 15) + 8
                            else:
                                sizes = 15
                            
                            fig.add_trace(go.Scattermapbox(
                                lat=subset_df['Lat'],
                                lon=subset_df['Long'],
                                mode='markers',
                                marker=dict(
                                    size=sizes,
                                    color=colors[i],
                                ),
                                text=subset_hover,
                                hovertemplate='%{text}<extra></extra>',
                                name=f"{val} (n={mask.sum()})"
                            ))
                else:
                    # Numerical coloring - single trace with colorscale
                    if size_column and size_column in df_clean.columns:
                        sizes = df_clean[size_column]
                        sizes = ((sizes - sizes.min()) / (sizes.max() - sizes.min()) * 15) + 8
                    else:
                        sizes = 15
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=df_clean['Lat'],
                        lon=df_clean['Long'],
                        mode='markers',
                        marker=dict(
                            size=sizes,
                            color=df_clean[color_column],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title=color_by)
                        ),
                        text=hover_data,
                        hovertemplate='%{text}<extra></extra>',
                        name='Bird Samples',
                        showlegend=False
                    ))
            
            # Configure map layout
            center_lat = df_clean['Lat'].mean()
            center_lon = df_clean['Long'].mean()
            
            fig.update_layout(
                mapbox=dict(
                    style=map_style,
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=5
                ),
                height=700,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left", 
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary Statistics
            st.header("📊 Data Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", len(df_clean))
            
            with col2:
                unique_locations = df_clean[['State', 'CityTown']].drop_duplicates()
                st.metric("Unique Locations", len(unique_locations))
            
            with col3:
                if 'inv_k3' in df_clean.columns:
                    unique_genotypes = df_clean['inv_k3'].nunique()
                    st.metric("Inversion Genotypes", unique_genotypes)
            
            with col4:
                states = df_clean['State'].nunique()
                st.metric("States/Provinces", states)
            
            # Detailed breakdowns
            col1, col2 = st.columns(2)
            
            with col1:
                if 'inv_k3' in df_clean.columns:
                    st.subheader("Inversion Genotype Distribution")
                    genotype_counts = df_clean['inv_k3'].value_counts()
                    fig_bar = px.bar(
                        x=genotype_counts.index,
                        y=genotype_counts.values,
                        labels={'x': 'Inversion Genotype', 'y': 'Count'},
                        title="Samples by Inversion Genotype"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                st.subheader("Geographic Distribution")
                state_counts = df_clean['State'].value_counts()
                fig_pie = px.pie(
                    values=state_counts.values,
                    names=state_counts.index,
                    title="Samples by State/Province"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Genetic metrics if available
            if 'MEAN_DEPTH' in df_clean.columns or 'coverage' in df_clean.columns:
                st.header("🧬 Genetic Data Quality")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'MEAN_DEPTH' in df_clean.columns:
                        fig_depth = px.histogram(
                            df_clean,
                            x='MEAN_DEPTH',
                            title="Distribution of Mean Sequencing Depth",
                            nbins=20
                        )
                        st.plotly_chart(fig_depth, use_container_width=True)
                
                with col2:
                    if 'coverage' in df_clean.columns:
                        fig_cov = px.histogram(
                            df_clean,
                            x='coverage',
                            title="Distribution of Coverage",
                            nbins=20
                        )
                        st.plotly_chart(fig_cov, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Please load data using one of the options in the sidebar")

# Instructions for GitHub setup
with st.expander("📋 How to set up your data on GitHub"):
    st.markdown("""
    ### Setting up your CSV data on GitHub:
    
    1. **Upload your CSV file** to your GitHub repository
    2. **Navigate to the file** in GitHub's web interface
    3. **Click "Raw"** to get the raw file URL
    4. **Copy the URL** - it should look like:
       ```
       https://raw.githubusercontent.com/username/repository/main/filename.csv
       ```
    5. **Paste the URL** in the sidebar input box
    
    ### Required columns in your CSV:
    - `BGP_ID`: Sample identifier
    - `Lat`: Latitude (decimal degrees)
    - `Long`: Longitude (decimal degrees) 
    - `inv_k3`: Inversion genotype
    - `State`: State/Province
    - `CityTown`: City/Town name
    
    ### Optional columns for enhanced visualization:
    - `Population`, `Climate`, `Sex`, `MEAN_DEPTH`, `coverage`, etc.
    
    ### Grid Overlay Features:
    - **Upload grid files**: GRD, GeoTIFF, ASCII, NetCDF formats supported
    - **Adjustable opacity**: Control transparency of grid overlay
    - **Color schemes**: Multiple color palettes for grid visualization
    - **Grid info**: Shows dimensions, value ranges, and bounds
    
    ### Common grid file uses:
    - **Elevation data**: Topographic information
    - **Climate data**: Temperature, precipitation, etc.
    - **Habitat data**: Land cover, vegetation indices
    - **Environmental variables**: Any spatially continuous data
    """)

# Footer
st.markdown("---")
st.markdown("🐦 Built with Streamlit & Plotly | Data visualization for genetic research")