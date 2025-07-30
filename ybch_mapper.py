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

def process_image_overlay(image_source):
    """Process image for overlay - handles both files and URLs"""
    try:
        from PIL import Image
        import base64
        import io
        import requests
        
        if isinstance(image_source, str):  # URL
            # Download image from URL
            response = requests.get(image_source)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
            else:
                st.error(f"Could not download image from URL: {response.status_code}")
                return None
        else:  # Uploaded file
            image = Image.open(image_source)
        
        # Convert to base64 for plotly
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

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
            
            # Range Map overlay option
            st.sidebar.header("🖼️ Range Map Overlay")
            image_overlay = st.sidebar.checkbox("Add range map overlay", value=False)
            
            image_file = None
            image_url = None
            
            if image_overlay:
                map_source = st.sidebar.radio(
                    "Range map source:",
                    ["Upload Custom Image", "Use Image URL"]
                )
                
                if map_source == "Upload Custom Image":
                    image_file = st.sidebar.file_uploader(
                        "Upload range map (.png, .jpg, .jpeg)",
                        type=['png', 'jpg', 'jpeg'],
                        help="Upload your range map image file"
                    )
                    
                elif map_source == "Use Image URL":
                    image_url = st.sidebar.text_input(
                        "Image URL:",
                        placeholder="https://raw.githubusercontent.com/user/repo/main/image.png",
                        help="Enter direct URL to your range map image"
                    )
                
                # Coordinate inputs
                if image_file or image_url:
                    st.sidebar.subheader("Image Coordinates")
                    st.sidebar.markdown("*Enter the geographic bounds of your range map:*")
                    
                    # Initialize session state for coordinates if not exists
                    if 'west_bound' not in st.session_state:
                        st.session_state.west_bound = -125.0
                    if 'east_bound' not in st.session_state:
                        st.session_state.east_bound = -73.9
                    if 'south_bound' not in st.session_state:
                        st.session_state.south_bound = 18.4
                    if 'north_bound' not in st.session_state:
                        st.session_state.north_bound = 51.5
                    
                    # Show data bounds for reference
                    if len(df_clean) > 0:
                        st.sidebar.info(f"Your data spans:\nLat: {df_clean['Lat'].min():.2f} to {df_clean['Lat'].max():.2f}\nLon: {df_clean['Long'].min():.2f} to {df_clean['Long'].max():.2f}")
                    
                    # Quick adjustment buttons FIRST (so they update session_state)
                    st.sidebar.markdown("**Quick Adjustments:**")
                    
                    # Position adjustments
                    col1, col2, col3 = st.sidebar.columns(3)
                    with col1:
                        if st.sidebar.button("⬅️ Move West"):
                            st.session_state.west_bound -= 5.0
                            st.session_state.east_bound -= 5.0
                    with col2:
                        if st.sidebar.button("⬆️ Move North"):
                            st.session_state.south_bound += 5.0
                            st.session_state.north_bound += 5.0
                    with col3:
                        if st.sidebar.button("➡️ Move East"):
                            st.session_state.west_bound += 5.0
                            st.session_state.east_bound += 5.0
                    
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        if st.sidebar.button("⬇️ Move South"):
                            st.session_state.south_bound -= 5.0
                            st.session_state.north_bound -= 5.0
                    
                    # Size adjustments
                    st.sidebar.markdown("**Resize Image:**")
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        if st.sidebar.button("🔍 Zoom In"):
                            # Make image smaller (zoom in)
                            center_lat = (st.session_state.north_bound + st.session_state.south_bound) / 2
                            center_lon = (st.session_state.east_bound + st.session_state.west_bound) / 2
                            lat_range = (st.session_state.north_bound - st.session_state.south_bound) * 0.8
                            lon_range = (st.session_state.east_bound - st.session_state.west_bound) * 0.8
                            st.session_state.north_bound = center_lat + lat_range/2
                            st.session_state.south_bound = center_lat - lat_range/2
                            st.session_state.east_bound = center_lon + lon_range/2
                            st.session_state.west_bound = center_lon - lon_range/2
                    with col2:
                        if st.sidebar.button("🔍 Zoom Out"):
                            # Make image bigger (zoom out)
                            center_lat = (st.session_state.north_bound + st.session_state.south_bound) / 2
                            center_lon = (st.session_state.east_bound + st.session_state.west_bound) / 2
                            lat_range = (st.session_state.north_bound - st.session_state.south_bound) * 1.2
                            lon_range = (st.session_state.east_bound - st.session_state.west_bound) * 1.2
                            st.session_state.north_bound = center_lat + lat_range/2
                            st.session_state.south_bound = center_lat - lat_range/2
                            st.session_state.east_bound = center_lon + lon_range/2
                            st.session_state.west_bound = center_lon - lon_range/2
                    
                    # Preset buttons for common ranges
                    st.sidebar.markdown("**Preset Ranges:**")
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        if st.sidebar.button("🌎 North America"):
                            st.session_state.west_bound = -130.0
                            st.session_state.east_bound = -60.0
                            st.session_state.south_bound = 20.0
                            st.session_state.north_bound = 60.0
                    with col2:
                        if st.sidebar.button("🎯 Fit to Data"):
                            margin = 3.0
                            st.session_state.west_bound = df_clean['Long'].min() - margin
                            st.session_state.east_bound = df_clean['Long'].max() + margin
                            st.session_state.south_bound = df_clean['Lat'].min() - margin
                            st.session_state.north_bound = df_clean['Lat'].max() + margin
                    
                    # Number inputs that reflect session_state values
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        west_bound = st.sidebar.number_input("West (min longitude)", value=st.session_state.west_bound, step=0.1, format="%.4f", key="west_input")
                        south_bound = st.sidebar.number_input("South (min latitude)", value=st.session_state.south_bound, step=0.1, format="%.4f", key="south_input")
                    with col2:
                        east_bound = st.sidebar.number_input("East (max longitude)", value=st.session_state.east_bound, step=0.1, format="%.4f", key="east_input")
                        north_bound = st.sidebar.number_input("North (max latitude)", value=st.session_state.north_bound, step=0.1, format="%.4f", key="north_input")
                    
                    # Update session_state when number inputs change
                    st.session_state.west_bound = west_bound
                    st.session_state.east_bound = east_bound
                    st.session_state.south_bound = south_bound
                    st.session_state.north_bound = north_bound
                    
                    image_opacity = st.sidebar.slider(
                        "Range map opacity",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.5,
                        step=0.1
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
                index=2
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
                ["Plotly (with image overlay)", "Mapbox (no image overlay)"]
            )
            
            # Process image overlay if selected
            image_data = None
            if image_overlay and (image_file or image_url):
                with st.spinner("Processing range map..."):
                    image_source = image_url if image_url else image_file
                    st.write(f"Debug: Processing image from: {image_source if isinstance(image_source, str) else 'uploaded file'}")
                    
                    image_base64 = process_image_overlay(image_source)
                    if image_base64:
                        st.success("✅ Range map loaded successfully")
                        image_data = {
                            'source': image_base64,
                            'bounds': (west_bound, south_bound, east_bound, north_bound),
                            'opacity': image_opacity
                        }
                        
                        # Show image info
                        with st.expander("🗺️ Range Map Info"):
                            st.write(f"**Image Bounds:** West={west_bound}, South={south_bound}, East={east_bound}, North={north_bound}")
                            st.write(f"**Your Data Bounds:** West={df_clean['Long'].min():.2f}, South={df_clean['Lat'].min():.2f}, East={df_clean['Long'].max():.2f}, North={df_clean['Lat'].max():.2f}")
                            st.write(f"Opacity: {image_opacity}")
                            if image_url:
                                st.write(f"Source: {image_url}")
                            st.write(f"Image data loaded: {len(image_base64) if image_base64 else 0} characters")
                            
                            # Show coordinate differences
                            west_diff = west_bound - df_clean['Long'].min()
                            east_diff = east_bound - df_clean['Long'].max()
                            south_diff = south_bound - df_clean['Lat'].min()
                            north_diff = north_bound - df_clean['Lat'].max()
                            
                            st.write("**Coordinate Differences (Image - Data):**")
                            st.write(f"West: {west_diff:.2f}°, East: {east_diff:.2f}°")
                            st.write(f"South: {south_diff:.2f}°, North: {north_diff:.2f}°")
                            
                            if abs(west_diff) > 2 or abs(east_diff) > 2 or abs(south_diff) > 2 or abs(north_diff) > 2:
                                st.warning("⚠️ Large coordinate differences detected. Try adjusting the image bounds.")
                                
                            # Suggested adjustments
                            st.write("**Suggested Image Bounds (based on your data + margin):**")
                            suggested_west = df_clean['Long'].min() - 5
                            suggested_east = df_clean['Long'].max() + 5
                            suggested_south = df_clean['Lat'].min() - 5
                            suggested_north = df_clean['Lat'].max() + 5
                            st.code(f"West: {suggested_west:.1f}, East: {suggested_east:.1f}, South: {suggested_south:.1f}, North: {suggested_north:.1f}")
                    else:
                        st.error("Failed to process range map")
            
            # Create the interactive map
            st.header("🗺️ Interactive Genetic Data Map")
            
            # Create the map figure
            fig = go.Figure()
            
            # Prepare hover data with your specified fields
            hover_data = []
            for _, row in df_clean.iterrows():
                hover_info = [
                    f"<b>{row['BGP_ID']}</b>",
                    f"Inversion: {row['inv_k3']}",
                    f"Location: {row['CityTown']}, {row['State']}",
                    f"Coordinates: {row['Lat']:.4f}, {row['Long']:.4f}"
                ]
                
                # Add population proportions if available
                if 'pop1' in row and pd.notna(row['pop1']):
                    hover_info.append(f"Pop1: {row['pop1']:.3f}")
                if 'pop2' in row and pd.notna(row['pop2']):
                    hover_info.append(f"Pop2: {row['pop2']:.3f}")
                
                # Add additional info if available
                if 'Population' in row and pd.notna(row['Population']):
                    hover_info.append(f"Population: {row['Population']}")
                if 'Sex' in row and pd.notna(row['Sex']):
                    hover_info.append(f"Sex: {row['Sex']}")
                if 'Climate' in row and pd.notna(row['Climate']):
                    hover_info.append(f"Climate: {row['Climate']}")
                if 'MEAN_DEPTH' in row and pd.notna(row['MEAN_DEPTH']):
                    hover_info.append(f"Mean Depth: {row['MEAN_DEPTH']:.2f}")
                if 'Month' in row and pd.notna(row['Month']) and 'Day' in row and pd.notna(row['Day']):
                    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    month = int(row['Month'])
                    day = int(row['Day'])
                    month_name = month_names[month] if 1 <= month <= 12 else str(month)
                    hover_info.append(f"Date: {month_name} {day}")
                
                hover_data.append("<br>".join(hover_info))
            
            # Handle coloring
            if color_column in df_clean.columns:
                if df_clean[color_column].dtype in ['object', 'category']:
                    # Categorical coloring
                    unique_vals = df_clean[color_column].unique()
                    # Use a nice color palette
                    colors = px.colors.qualitative.Set1
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
                                        (df_clean[size_column].max() - df_clean[size_column].min()) * 25) + 15
                            else:
                                sizes = 15
                            
                            # Choose plot type based on map style
                            if map_style == "Plotly (with image overlay)":
                                fig.add_trace(go.Scatter(
                                    x=subset_df['Long'],
                                    y=subset_df['Lat'],
                                    mode='markers',
                                    marker=dict(
                                        size=sizes,
                                        color=colors[i],
                                    ),
                                    text=subset_hover,
                                    hovertemplate='<span style="color: black;">%{text}</span><extra></extra>',
                                    name=f"{val} (n={mask.sum()})"
                                ))
                            else:
                                fig.add_trace(go.Scattermapbox(
                                    lat=subset_df['Lat'],
                                    lon=subset_df['Long'],
                                    mode='markers',
                                    marker=dict(
                                        size=sizes,
                                        color=colors[i],
                                    ),
                                    text=subset_hover,
                                    hovertemplate='<span style="color: black;">%{text}</span><extra></extra>',
                                    name=f"{val} (n={mask.sum()})"
                                ))
                else:
                    # Numerical coloring - single trace with colorscale
                    if size_column and size_column in df_clean.columns:
                        sizes = df_clean[size_column]
                        sizes = ((sizes - sizes.min()) / (sizes.max() - sizes.min()) * 25) + 15
                    else:
                        sizes = 15
                    
                    # Choose plot type based on map style
                    if map_style == "Plotly (with image overlay)":
                        fig.add_trace(go.Scatter(
                            x=df_clean['Long'],
                            y=df_clean['Lat'],
                            mode='markers',
                            marker=dict(
                                size=sizes,
                                color=df_clean[color_column],
                                colorscale='Turbo',
                                showscale=True,
                                colorbar=dict(title=color_by)
                            ),
                            text=hover_data,
                            hovertemplate='<span style="color: black;">%{text}</span><extra></extra>',
                            name='Bird Samples',
                            showlegend=False
                        ))
                    else:
                        fig.add_trace(go.Scattermapbox(
                            lat=df_clean['Lat'],
                            lon=df_clean['Long'],
                            mode='markers',
                            marker=dict(
                                size=sizes,
                                color=df_clean[color_column],
                                colorscale='Turbo',
                                showscale=True,
                                colorbar=dict(title=color_by)
                            ),
                            text=hover_data,
                            hovertemplate='<span style="color: black;">%{text}</span><extra></extra>',
                            name='Bird Samples',
                            showlegend=False
                        ))
            
            # Configure map layout (basic settings)
            center_lat = df_clean['Lat'].mean()
            center_lon = df_clean['Long'].mean()
            
            fig.update_layout(
                height=700,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left", 
                    x=0.01,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    font_color="black",
                    bordercolor="black",
                    font_size=12
                )
            )
            
            # Add range map overlay after mapbox is configured
            if image_data:
                st.write(f"Debug: Adding image overlay after map setup")
                
                if map_style == "Plotly (with image overlay)":
                    # Use regular Plotly with image overlay (works better)
                    try:
                        fig.add_layout_image(
                            source=image_data['source'],
                            xref="x",
                            yref="y",
                            x=image_data['bounds'][0],  # west
                            y=image_data['bounds'][1],  # south
                            sizex=image_data['bounds'][2] - image_data['bounds'][0],  # width
                            sizey=image_data['bounds'][3] - image_data['bounds'][1],  # height
                            opacity=image_data['opacity'],
                            layer="below"
                        )
                        st.write("Debug: Image overlay added successfully to regular Plotly")
                        
                        # Configure regular Plotly layout
                        fig.update_layout(
                            xaxis=dict(
                                range=[image_data['bounds'][0] - 2, image_data['bounds'][2] + 2],
                                showgrid=True,
                                title="Longitude"
                            ),
                            yaxis=dict(
                                range=[image_data['bounds'][1] - 2, image_data['bounds'][3] + 2],
                                showgrid=True,
                                title="Latitude"
                            )
                        )
                        
                    except Exception as e:
                        st.error(f"Error adding image to regular Plotly: {e}")
                else:
                    # Mapbox layout without image overlay
                    st.warning("Image overlay not supported with mapbox style")
                    fig.update_layout(
                        mapbox=dict(
                            style="open-street-map",
                            center=dict(lat=center_lat, lon=center_lon),
                            zoom=5
                        )
                    )
            else:
                # Configure layout based on map style
                if map_style == "Plotly (with image overlay)":
                    fig.update_layout(
                        xaxis=dict(
                            range=[df_clean['Long'].min() - 2, df_clean['Long'].max() + 2],
                            showgrid=True,
                            title="Longitude"
                        ),
                        yaxis=dict(
                            range=[df_clean['Lat'].min() - 2, df_clean['Lat'].max() + 2],
                            showgrid=True,
                            title="Latitude"
                        )
                    )
                else:
                    fig.update_layout(
                        mapbox=dict(
                            style="open-street-map",
                            center=dict(lat=center_lat, lon=center_lon),
                            zoom=5
                        )
                    )
            
            st.plotly_chart(fig, use_container_width=True, key="main_map")
            
            # STRUCTURE-style Admixture Plot
            if 'pop1' in df_clean.columns and 'pop2' in df_clean.columns:
                st.header("🧬 Population Structure (ADMIXTURE)")
                
                # Sort samples for better visualization (optional)
                sort_option = st.selectbox(
                    "Sort samples by:",
                    ["Original order", "Pop1 proportion", "Pop2 proportion", "Geographic (Longitude)", "Geographic (Latitude)"],
                    index=1,
                    help="Choose how to order samples in the structure plot"
                )
                
                df_plot = df_clean.copy()
                if sort_option == "Pop1 proportion":
                    df_plot = df_plot.sort_values('pop1', ascending=False)
                elif sort_option == "Pop2 proportion":
                    df_plot = df_plot.sort_values('pop2', ascending=False)
                elif sort_option == "Geographic (Longitude)":
                    df_plot = df_plot.sort_values('Long')
                elif sort_option == "Geographic (Latitude)":
                    df_plot = df_plot.sort_values('Lat')
                
                # Reset index to get position for x-axis
                df_plot = df_plot.reset_index(drop=True)
                df_plot['x_position'] = range(len(df_plot))
                
                # Create STRUCTURE plot
                fig_structure = go.Figure()
                
                # Add pop1 (bottom layer)
                fig_structure.add_trace(go.Bar(
                    x=df_plot['x_position'],
                    y=df_plot['pop1'],
                    name='Population 1',
                    marker_color='#004488',
                    hovertemplate='<span style="color: #000000;"><b>%{customdata[0]}</b></span><br>' +
                                 '<span style="color: #000000;">Pop1: %{y:.3f}</span><br>' +
                                 '<span style="color: #000000;">Pop2: %{customdata[1]:.3f}</span><br>' +
                                 '<span style="color: #000000;">Location: %{customdata[2]}, %{customdata[3]}</span><br>' +
                                '<span style="color: #000000;">Coordinates: %{customdata[4]:.4f}, %{customdata[5]:.4f}</span><extra></extra>',
                    customdata=df_plot[['BGP_ID', 'pop2', 'CityTown', 'State', 'Lat', 'Long']].values
                ))
                
                # Add pop2 (stacked on top)
                fig_structure.add_trace(go.Bar(
                    x=df_plot['x_position'],
                    y=df_plot['pop2'],
                    base=df_plot['pop1'],
                    name='Population 2',
                    marker_color='#5aae61',
                    hovertemplate='<span style="color: #000000;"><b>%{customdata[0]}</b></span><br>' +
                                 '<span style="color: #000000;">Pop1: %{y:.3f}</span><br>' +
                                 '<span style="color: #000000;">Pop2: %{customdata[1]:.3f}</span><br>' +
                                 '<span style="color: #000000;">Location: %{customdata[2]}, %{customdata[3]}</span><br>' +
                                '<span style="color: #000000;">Coordinates: %{customdata[4]:.4f}, %{customdata[5]:.4f}</span><extra></extra>',
                    customdata=df_plot[['BGP_ID', 'pop1', 'CityTown', 'State', 'Lat', 'Long']].values
                ))
                
                # Update layout for STRUCTURE plot
                fig_structure.update_layout(
                    title="Population Admixture Proportions",
                    xaxis_title="Individual Samples",
                    yaxis_title="Ancestry Proportion",
                    barmode='stack',
                    height=400,
                    showlegend=True,
                    xaxis=dict(
                        showticklabels=False,
                        showgrid=False
                    ),
                    yaxis=dict(
                        range=[0, 1],
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    ),
                    plot_bgcolor='white',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        font=dict(
                            color="black",
                            size=12,
                            family="Arial"
                        )
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_color="black",
                        bordercolor="black",
                        font_size=12
                    )
                )
                
                st.plotly_chart(fig_structure, use_container_width=True, key="structure_plot")
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_pop1 = df_clean['pop1'].mean()
                    st.metric("Average Pop1", f"{avg_pop1:.3f}")
                
                with col2:
                    avg_pop2 = df_clean['pop2'].mean() 
                    st.metric("Average Pop2", f"{avg_pop2:.3f}")
                
                with col3:
                    # Count highly admixed individuals (0.3 < pop1 < 0.7)
                    admixed = df_clean[(df_clean['pop1'] > 0.3) & (df_clean['pop1'] < 0.7)]
                    st.metric("Admixed Individuals", f"{len(admixed)}")
                
                # Optional: Add population assignment by geographic region
                if 'State' in df_clean.columns:
                    st.subheader("📍 Population Structure by Geographic Region")
                    
                    # Calculate average admixture by state
                    state_summary = df_clean.groupby('State').agg({
                        'pop1': ['mean', 'std', 'count'],
                        'pop2': ['mean', 'std']
                    }).round(3)
                    
                    # Flatten column names
                    state_summary.columns = ['Pop1_Mean', 'Pop1_Std', 'Sample_Count', 'Pop2_Mean', 'Pop2_Std']
                    state_summary = state_summary.reset_index()
                    
                    # Display as interactive table
                    st.dataframe(
                        state_summary,
                        use_container_width=True,
                        hide_index=True
                    )
            
            else:
                st.info("💡 Pop1 and Pop2 columns not found - upload data with admixture proportions to see STRUCTURE plot")
            
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
    
    ### Range Map Features:
    - **Default range map**: Automatically loads YBCH genoscape
    - **Custom upload**: Upload your own range map image
    - **URL support**: Link to images hosted online
    - **Adjustable opacity**: Control transparency of range overlay
    """)

# Footer
st.markdown("---")
st.markdown("🐦 Built with Streamlit & Plotly | Data visualization for genetic research")