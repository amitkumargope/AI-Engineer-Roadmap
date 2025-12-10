# AI-Engineer-Roadmap
🚀 A 20-week journey mastering Data Science, Machine Learning, Deep Learning, Computer Vision, MLOps, and Generative AI. This repository documents my complete learning plan, projects, and portfolio as I transition into advanced AI Engineering roles across industries like semiconductors, healthcare, materials science, and industrial automation.
# 🏭 Intelligent Warehouse Simulation System

## Overview
This is an **interactive warehouse order picking simulation** with animated visualizations, designed to be easy to use even for non-technical users. The system uses real warehouse data to demonstrate intelligent pathfinding and order fulfillment processes.

## 🎯 Features

### 1. **Warehouse Visualization**
- 🗺️ 2D floor layouts showing all storage locations
- 🌐 3D warehouse view with multiple floors
- 📍 Navigation points and corridor mapping
- 🎨 Color-coded storage zones

### 2. **Intelligent Pathfinding**
- 🧠 A* algorithm for optimal route calculation
- 🚀 Nearest neighbor heuristic for multi-pick optimization
- 📏 Automatic distance calculation
- 🛣️ Smart navigation through warehouse corridors

### 3. **Animated Order Picking**
- 🎬 Real-time animation of picker movement
- 📦 Visual highlighting of items to pick
- 📊 Live statistics during picking
- 🔄 Customizable animation speed

### 4. **Interactive Dashboard**
- 🎮 Easy-to-use controls (no coding required!)
- 📋 Wave selection dropdown
- 🏢 Floor selector
- ⚙️ Adjustable parameters
- 🔘 One-click operation

### 5. **Performance Analytics**
- 📈 Operator efficiency comparison
- 📊 Distance metrics and statistics
- 🏆 Best/worst performing waves
- 📉 Trend analysis

## 🚀 Quick Start

### For Non-Technical Users

1. **Open the Notebook**
   - Open `Intelligent_warehouse.ipynb` in Jupyter/VS Code

2. **Run All Cells**
   - Click "Run All" in the menu (or press Ctrl+Shift+Enter)
   - Wait for all cells to execute (should take ~10 seconds)

3. **Use the Interactive Dashboard**
   - Scroll to **Section 5: Interactive Control Panel**
   - Use the dropdown menus to select:
     - Wave number (different order picking sessions)
     - Floor level (1-4)
     - Maximum picks to show
   - Click the buttons:
     - **▶ Animate Picking** - Watch the animation
     - **🗺️ Show Floor Layout** - View warehouse layout
     - **📊 Compare Waves** - Compare different sessions

4. **Understand the Visualizations**
   - 🟡 Yellow boxes = Items to pick
   - 🔴 Red star = Current picker location
   - 🟢 Green dashed line = Planned route
   - 🔵 Blue solid line = Completed path
   - ⭐ Red stars = Navigation waypoints

### For Technical Users

#### Installation
All required packages are already imported:
- pandas, numpy (data processing)
- matplotlib (visualization)
- ipywidgets (interactive controls)
- seaborn (styling)

#### Key Classes

**WarehouseNavigator**
```python
navigator = WarehouseNavigator(storage_locations, support_points)
route, distance = navigator.calculate_picking_route(pick_locations)
```

**OrderPickingAnimator**
```python
animator = OrderPickingAnimator(navigator, storage_locations, support_points)
animator.animate_picking_wave(wave_number=43175, max_picks=10)
```

## 📊 Dataset Structure

### Files Used
- `Storage_Location.csv` - 2,292 storage positions with X,Y,Z coordinates
- `Customer_Order.csv` - 122,370 order line items
- `Picking_Wave.csv` - 215,192 individual picks
- `Support_Points_Navigation.csv` - 44 navigation waypoints
- `Product.csv` - 208 unique products

### Key Metrics
- **Warehouse Dimensions**: 580m × 1440m
- **Floors**: 4 levels (Z = 1 to 4)
- **Navigation Corridors**: LC (vertical), CC (horizontal)
- **Operators**: 2 operators tracked

## 🎓 Understanding the System

### Pathfinding Algorithm
The system uses **A* (A-Star)** pathfinding:
1. Finds nearest navigation point to start location
2. Calculates optimal path through navigation network
3. Uses Euclidean distance as heuristic
4. Returns complete waypoint sequence

### Route Optimization
For multiple picks, uses **Nearest Neighbor**:
1. Start at depot (entrance)
2. Pick nearest unpicked item
3. Navigate to that location
4. Repeat until all items collected
5. Return to depot

### Animation System
- Uses matplotlib FuncAnimation
- Updates picker position frame-by-frame
- Shows real-time statistics
- Configurable speed (50-500ms per frame)

## 📈 Example Use Cases

### 1. Training & Education
- Train new warehouse staff
- Visualize optimal picking paths
- Demonstrate warehouse layout

### 2. Performance Analysis
- Compare operator efficiency
- Identify bottlenecks
- Optimize storage allocation

### 3. Process Improvement
- Test different picking strategies
- Evaluate layout changes
- Simulate peak periods

### 4. Research & Development
- Study warehouse optimization algorithms
- Test new pathfinding methods
- Analyze picking patterns

## 🎨 Customization

### Change Animation Speed
```python
# In the interactive dashboard
animation_speed.value = 100  # milliseconds per frame
```

### Adjust Number of Picks
```python
max_picks_slider.value = 20  # show more picks
```

### Visualize Specific Floor
```python
plot_warehouse_floor(floor_level=1)
```

### Compare Different Waves
```python
animator.create_multi_wave_comparison([43175, 43176, 43177])
```

## 🔧 Troubleshooting

### Animation Not Showing
- Make sure you're running in Jupyter/VS Code notebook
- Try using `%matplotlib widget` for interactive plots
- Check that all cells have been executed

### Missing Data
- Ensure all CSV files are in `Order Picking Dataset/` folder
- Check file paths are correct
- Verify CSV encoding (should be UTF-8)

### Slow Performance
- Reduce `max_picks` value
- Use fewer waves for comparison
- Close other applications

### Widget Not Interactive
- Install ipywidgets: `pip install ipywidgets`
- Enable widgets in Jupyter: `jupyter nbextension enable --py widgetsnbextension`

## 📚 Code Structure

```
Intelligent_warehouse.ipynb
├── Section 1: Library Imports
├── Section 2: Data Loading
├── Section 3: Warehouse Visualization
│   ├── 2D floor plots
│   └── 3D warehouse view
├── Section 4: Pathfinding System
│   ├── WarehouseNavigator class
│   └── A* algorithm implementation
├── Section 5: Animation System
│   ├── OrderPickingAnimator class
│   └── Multi-wave comparison
├── Section 6: Interactive Dashboard
│   └── Widget controls
└── Section 7: Examples & Documentation
```

## 🎯 Future Enhancements (Ideas for Extension)

### Machine Learning Integration
- [ ] Predict picking times using historical data
- [ ] Optimize storage allocation based on frequency
- [ ] Forecast demand patterns
- [ ] Anomaly detection for unusual patterns

### Real-time Features
- [ ] Multi-picker simulation (avoid collisions)
- [ ] Dynamic re-routing based on conditions
- [ ] Live tracking integration
- [ ] Voice guidance simulation

### Advanced Analytics
- [ ] Heat maps of warehouse activity
- [ ] Bottleneck identification
- [ ] Seasonal pattern analysis
- [ ] ABC analysis visualization

### Integration Options
- [ ] Export routes to WMS systems
- [ ] API for external systems
- [ ] Mobile app interface
- [ ] VR/AR warehouse walkthrough

## 👥 Target Audience

✅ **Warehouse Managers** - Optimize operations
✅ **Operations Analysts** - Performance analysis
✅ **Trainers** - Staff education
✅ **Students** - Learning warehouse management
✅ **Researchers** - Algorithm development
✅ **Developers** - System integration

## 📝 License & Usage
This is an educational and research tool. Feel free to modify and extend for your needs.

## 🆘 Support
For questions or issues:
1. Check the troubleshooting section
2. Review the Quick Start Guide
3. Examine the example demonstrations in Section 6
4. Review comments in the code

## 🎉 Credits
Built using:
- Python 3.11+
- Matplotlib for visualization
- Pandas for data processing
- ipywidgets for interactivity
- Real warehouse data from footwear manufacturing

---

**Happy Simulating! 🏭📦🚀**
