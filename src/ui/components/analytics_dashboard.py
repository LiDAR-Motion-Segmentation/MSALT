import pyqtgraph as pg
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QSpinBox, QGridLayout
)
from collections import defaultdict

class AnalyticsDashboard(QDialog):
    def __init__(self, annotation_manager, max_frames: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MSALT QA Analytics & Telemetry")
        self.setMinimumSize(1000, 700)
        
        self.annotation_manager = annotation_manager
        self.max_frames = max_frames
        
        # Apply dark theme for that "Grafana" look
        pg.setConfigOption('background', '#111111')
        pg.setConfigOption('foreground', '#d3d3d3')
        
        self._setup_ui()
        self._compute_and_draw() # Initial draw
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Top Control Bar
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Analyze Range:"))
        
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, self.max_frames)
        self.spin_start.setValue(0)
        
        self.spin_end = QSpinBox()
        self.spin_end.setRange(0, self.max_frames)
        self.spin_end.setValue(self.max_frames)
        
        self.btn_refresh = QPushButton("Compute Metrics")
        self.btn_refresh.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_refresh.clicked.connect(self._compute_and_draw)
        
        ctrl_layout.addWidget(QLabel("Start Frame:"))
        ctrl_layout.addWidget(self.spin_start)
        ctrl_layout.addWidget(QLabel("End Frame:"))
        ctrl_layout.addWidget(self.spin_end)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.btn_refresh)
        
        layout.addLayout(ctrl_layout)
        
        # graphana grid
        grid = QGridLayout()
        
        # completeness plot
        self.plt_completeness = pg.PlotWidget(title="Completeness (Objects per Frame)")
        self.plt_completeness.setLabel('left', 'Total Objects')
        self.plt_completeness.setLabel('bottom', 'Frame Index')
        self.plt_completeness.showGrid(x=True, y=True, alpha=0.3)
        grid.addWidget(self.plt_completeness, 0, 0)
        
        # class distribution plot
        self.plt_classes = pg.PlotWidget(title="Class Distribution (Ontology)")
        self.plt_classes.setLabel('left', 'Total Count')
        grid.addWidget(self.plt_classes, 0, 1)
        
        # Consistency (Lifespan)
        self.plt_lifespan = pg.PlotWidget(title="Consistency (Track Lifespan in Frames)")
        self.plt_lifespan.setLabel('left', 'Number of IDs')
        self.plt_lifespan.setLabel('bottom', 'Frames Survived')
        grid.addWidget(self.plt_lifespan, 1, 0)
        
        # Validity (Box Volume Outliers)
        self.plt_validity = pg.PlotWidget(title="Validity (Box Volumes)")
        self.plt_validity.setLabel('left', 'Volume (m³)')
        self.plt_validity.setLabel('bottom', 'Track ID')
        grid.addWidget(self.plt_validity, 1, 1)
        
        layout.addLayout(grid)
        
    def _compute_and_draw(self):
        start = self.spin_start.value()
        end = self.spin_end.value()
        
        # Clear old data
        self.plt_completeness.clear()
        self.plt_classes.clear()
        self.plt_lifespan.clear()
        self.plt_validity.clear()
        
        # Data structures for metrics
        frames = []
        counts = []
        
        class_counts = defaultdict(int)
        id_lifespans = defaultdict(int)
        id_volumes = {}
        
        for f_idx in range(start, end + 1):
            boxes = self.annotation_manager.get_boxes(f_idx)
            frames.append(f_idx)
            counts.append(len(boxes))
            
            for box in boxes:
                class_counts[box.label] += 1
                id_lifespans[box.track_id] += 1
                
                # Calculate physical volume for validity checks
                volume = box.dx * box.dy * box.dz
                if box.track_id not in id_volumes or volume > id_volumes[box.track_id]:
                    id_volumes[box.track_id] = volume # Store max volume seen for this ID
                    
        if not frames:
            return
        
        # draw completeness (line chart)
        pen = pg.mkPen(color='#00FF00', width=2)
        self.plt_completeness.plot(frames, counts, pen=pen, fillLevel=0, brush=(0,255,0,50))
        
        # draw class distribution (Bar Chart)
        if class_counts:
            labels = list(class_counts.keys())
            x = np.arange(len(labels))
            y = list(class_counts.values())
                    
            bg = pg.BarGraphItem(x=x, height=y, width=0.6, brush='b')
            self.plt_classes.addItem(bg)
            
            # Add text labels to X-axis
            ticks = [list(zip(x, labels))]
            self.plt_classes.getAxis('bottom').setTicks(ticks)
            
        # draw consistency (lifespan histogram)
        if id_lifespans:
            lifespans = list(id_lifespans.values())
            
            # simple histogram calculation
            y, x = np.histogram(lifespans, bins=min(10, max(1, len(set(lifespans)))))
            bg_life = pg.BarGraphItem(x0=x[:-1], x1=x[1:], height=y, brush='y')
            self.plt_lifespan.addItem(bg_life)
            
        # draw validity (Scatter Plot)
        if id_volumes:
            ids = list(id_volumes.keys())
            vols = list(id_volumes.values())
            
            # Color extreme outliers red
            mean_vol = np.mean(vols)
            std_vol = np.std(vols)
            colors = [
                pg.mkBrush('r') if abs(v - mean_vol) > (2 * std_vol) else pg.mkBrush('c') 
                for v in vols
            ]
            
            scatter = pg.ScatterPlotItem(x=ids, y=vols, size=8, brush=colors)
            self.plt_validity.addItem(scatter)