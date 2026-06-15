# SAM 3 YOLO Auto-Labeler Setup Guide (Google Colab)

This guide turns your Google Colab instance into an on-demand SAM 3 inference server and annotation tool. It features a completely revamped, framework-less, lightning-fast WebUI inspired by brutalist web design principles. Google Drive acts as the single source of truth for your data and class ontology.

### Prerequisites:
1. **Hugging Face Token:** You must agree to the SAM 3 license at [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) and generate an Access Token.
2. **Ngrok Token:** Get your authtoken from [ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken).
3. **Colab Secrets:** In the left sidebar of Colab (🔑 icon), add `HF_TOKEN` and `NGROK_TOKEN`. Ensure "Notebook access" is toggled ON for both.

---

### Cell 1: Install Dependencies
Open a new Google Colab notebook, set the runtime to **T4 GPU**, and run this cell. 
*(Note: Colab may prompt you to **Restart Session** at the bottom of the output after this runs. Please click it before proceeding).*

```bash
# Create templates directory for the web UI
!mkdir -p templates

# Clone SAM 3 repository and install
!git clone https://github.com/facebookresearch/sam3.git
%cd sam3
!pip install -e .

# Install Flask, CORS, and Ngrok
!pip install flask flask-cors pyngrok opencv-python-headless
```

---

### Cell 2: Mount Google Drive
This connects your Google Drive so the dataset and configuration are saved permanently.

```python
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive mounted successfully!")
```

---

### Cell 3: Create the Frontend UI (`index.html`)
*Note: `%%writefile` must be the very first line in the cell.*

```html
%%writefile templates/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAM 3 Lightning Labeler</title>
    <style>
        :root { --bg: #121212; --panel: #1e1e1e; --text: #e0e0e0; --accent: #007bff; --danger: #dc3545; --success: #28a745; --border: #333; }
        body { margin: 0; font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; }
        
        nav { width: 60px; background: #000; display: flex; flex-direction: column; align-items: center; padding-top: 20px; gap: 30px; border-right: 1px solid var(--border);}
        .nav-btn { cursor: pointer; background: none; border: none; font-size: 24px; opacity: 0.4; transition: 0.2s; padding: 10px; border-radius: 8px;}
        .nav-btn.active, .nav-btn:hover { opacity: 1; background: #333; }
        
        .view-panel { flex-grow: 1; display: none; overflow-y: auto; box-sizing: border-box; }
        .view-panel.active { display: flex; flex-direction: column; }
        .padded-view { padding: 30px; max-width: 1200px; margin: 0 auto; width: 100%; }
        
        #view-studio { flex-direction: row; padding: 0; }
        .canvas-container { flex-grow: 1; position: relative; background: #050505; display: flex; align-items: center; justify-content: center; overflow: hidden;}
        canvas { max-width: 100%; max-height: 100%; cursor: crosshair; }
        aside { width: 340px; background: var(--panel); padding: 20px; display: flex; flex-direction: column; gap: 15px; border-left: 1px solid var(--border); overflow-y: auto;}
        
        h2 { margin-top: 0; border-bottom: 2px solid var(--border); padding-bottom: 10px; }
        button { padding: 10px 15px; border: none; border-radius: 4px; font-weight: bold; cursor: pointer; background: #444; color: white; transition: 0.2s; }
        button:hover { filter: brightness(1.2); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-accent { background: var(--accent); }
        .btn-success { background: var(--success); }
        .btn-danger { background: var(--danger); }
        .btn-outline { background: transparent; border: 1px solid #666; color: #ccc; padding: 4px 8px; font-size: 11px;}
        input[type="text"], input[type="color"] { padding: 8px; background: #2a2a2a; border: 1px solid #444; color: white; border-radius: 4px; width: 100%; box-sizing: border-box;}
        
        .sub-tabs { display: flex; gap: 10px; margin-bottom: 15px; }
        .sub-tab-btn { background: transparent; border: 1px solid var(--border); color: #888; border-radius: 20px; padding: 6px 15px; }
        .sub-tab-btn.active { background: #333; color: white; border-color: #555; }
        
        /* Explorer / Gallery styles */
        .breadcrumb { font-size: 14px; margin-bottom: 15px; background: #222; padding: 10px; border-radius: 6px; display: flex; align-items: center; gap: 8px; }
        .breadcrumb span { cursor: pointer; color: var(--accent); }
        .breadcrumb span:hover { text-decoration: underline; }
        
        .compact-gallery { display: flex; flex-direction: column; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
        .compact-item { display: flex; align-items: center; justify-content: space-between; padding: 10px 15px; background: #222; border-bottom: 1px solid var(--border); transition: background 0.1s;}
        .compact-item:hover { background: #2a2a2a; }
        .compact-item.folder { cursor: pointer; }
        .compact-item:last-child { border-bottom: none; }
        .compact-left { display: flex; align-items: center; gap: 15px; }
        
        #bulkActionBar { background: var(--accent); color: white; padding: 10px 15px; border-radius: 6px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;}
        
        .badge { font-size: 10px; padding: 3px 8px; border-radius: 12px; font-weight: bold; text-transform: uppercase;}
        .badge.approved { background: #1e4620; color: #5cb85c; }
        .badge.pending { background: #463c1e; color: #f0ad4e; }
        
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid var(--border); }
        th { background: #222; }

        .loader-overlay { position: absolute; inset: 0; background: rgba(0,0,0,0.85); display: none; flex-direction: column; align-items: center; justify-content: center; z-index: 10; text-align: center; }
        .spinner { border: 4px solid #333; border-top: 4px solid var(--accent); border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin-bottom: 15px;}
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .progress-bar { width: 300px; height: 10px; background: #333; border-radius: 5px; margin-top: 20px; overflow: hidden; }
        .progress-fill { height: 100%; background: var(--accent); width: 0%; transition: 0.2s; }

        #toast { position: fixed; bottom: 20px; right: 20px; background: var(--panel); padding: 15px 20px; border-radius: 4px; box-shadow: 0 5px 15px rgba(0,0,0,0.5); transform: translateY(150%); transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); z-index: 1000; border-left: 5px solid var(--success); font-weight: bold;}
        #toast.show { transform: translateY(0); }

        .class-row { display: flex; align-items: center; gap: 10px; background: #2a2a2a; border-radius: 4px; padding: 4px 8px; border: 1px solid #333;}
        .class-btn { flex-grow: 1; display: flex; align-items: center; justify-content: space-between; background: transparent; border: none; padding: 6px; text-align: left;}
        .class-row.selected { border-color: white; background: #333; }
        .color-box { width: 14px; height: 14px; border-radius: 3px; display: inline-block; border: 1px solid rgba(255,255,255,0.2);}
    </style>
</head>
<body>

    <nav>
        <button class="nav-btn active" onclick="switchView('studio')" title="Annotation Studio">🎨</button>
        <button class="nav-btn" onclick="switchView('data')" title="Data Management">📁</button>
        <button class="nav-btn" onclick="switchView('ontology')" title="Classes & Prompts">🏷️</button>
    </nav>

    <!-- 1. Studio View -->
    <main id="view-studio" class="view-panel active">
        <div class="canvas-container">
            <canvas id="editorCanvas"></canvas>
            
            <div id="samLoader" class="loader-overlay">
                <div class="spinner"></div>
                <div style="font-weight:bold; letter-spacing: 1px;">SAM 3 INFERENCING...</div>
            </div>

            <div id="bulkLoader" class="loader-overlay">
                <div class="spinner"></div>
                <h2 style="margin: 10px 0;">🤖 Bulk Processing Queue</h2>
                <div id="bulkStatusText" style="color: #aaa; margin-bottom: 10px;">Initializing...</div>
                <div class="progress-bar"><div id="bulkProgressFill" class="progress-fill"></div></div>
                <div id="bulkEtaText" style="margin-top: 15px; font-weight: bold; color: var(--accent);">ETA: --:--</div>
                <div style="margin-top: 20px; font-size: 12px; color: #888;">Note: Annotations are being saved directly to Google Drive.</div>
            </div>
        </div>
        <aside>
            <div style="background: #111; padding: 10px; border-radius: 6px; border: 1px solid var(--border);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <div style="font-size: 11px; color: #888; text-transform: uppercase;">Active Queue</div>
                    <div style="font-size: 11px; color: #888;" id="queueCount">0 / 0</div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <button onclick="navQueue(-1)" class="btn-outline">◀ Prev</button>
                    <div id="currentFilename" style="font-weight: bold; font-size: 12px; word-break: break-all; text-align: center; margin: 0 5px;">No files loaded</div>
                    <button onclick="navQueue(1)" class="btn-outline">Next ▶</button>
                </div>
            </div>
            
            <button onclick="startBulkRun()" class="btn-accent" style="padding: 12px;">🤖 Bulk Process Entire Queue</button>
            <hr style="border-color: #333; margin: 0;">
            
            <div style="flex-grow: 1; display: flex; flex-direction: column;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="font-size: 11px; color: #888; text-transform: uppercase;">SAM 3 Targeting</div>
                    <div style="display: flex; gap: 5px;">
                        <button class="btn-outline" onclick="toggleAllSAM(true)">All</button>
                        <button class="btn-outline" onclick="toggleAllSAM(false)">None</button>
                    </div>
                </div>
                <div id="activeClassesList" style="display: flex; flex-direction: column; gap: 6px; overflow-y: auto;"></div>
            </div>

            <hr style="border-color: #333; margin: 0;">
            <button onclick="runSAM3Single()" id="btnSam" style="padding: 12px;">🎯 Label Current File</button>
            <button onclick="clearPolygons()" class="btn-danger">🗑️ Clear Polygons</button>
            <button onclick="saveAndNext()" id="btnSave" class="btn-success" style="padding: 15px; margin-top: 10px;">💾 Save & Next</button>
        </aside>
    </main>

    <!-- 2. Data View -->
    <main id="view-data" class="view-panel">
        <div class="padded-view">
            <h2>Data Management</h2>
            
            <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 15px;">
                <div class="sub-tabs" style="margin-bottom: 0;">
                    <button id="tab-pending" class="sub-tab-btn active" onclick="setGalleryFilter('pending')">Pending Views</button>
                    <button id="tab-approved" class="sub-tab-btn" onclick="setGalleryFilter('approved')">Approved Views</button>
                    <button id="tab-all" class="sub-tab-btn" onclick="setGalleryFilter('all')">View All</button>
                </div>
                <div style="display: flex; gap: 10px;">
                    <input type="file" id="fileUpload" multiple accept="image/*" style="display: none;" onchange="handleUpload(event)">
                    <input type="file" id="folderUpload" webkitdirectory directory multiple style="display: none;" onchange="handleUpload(event)">
                    <input type="file" id="zipUpload" accept=".zip" style="display: none;" onchange="handleZipUpload(event)">
                    
                    <button onclick="document.getElementById('fileUpload').click()">📄 Upload Files</button>
                    <button onclick="document.getElementById('folderUpload').click()">📁 Upload Folder</button>
                    <button onclick="document.getElementById('zipUpload').click()">📦 Upload ZIP</button>
                    <button onclick="refreshGallery(true)" class="btn-accent">🔄 Refresh</button>
                </div>
            </div>

            <div class="breadcrumb" id="breadcrumbNav"></div>

            <div id="bulkActionBar" style="display: none;">
                <div><strong id="bulkCount">0</strong> items selected</div>
                <div>
                    <button onclick="loadSelectedIntoStudio()" style="background: transparent; border: 1px solid white; padding: 6px 12px; margin-right: 10px;">🎨 Open in Studio</button>
                    <button onclick="bulkDelete()" style="background: transparent; border: 1px solid white; padding: 6px 12px;">🗑️ Delete Selected</button>
                </div>
            </div>
            
            <div class="compact-gallery" id="galleryContainer"></div>
        </div>
    </main>

    <!-- 3. Ontology View -->
    <main id="view-ontology" class="view-panel">
        <div class="padded-view">
            <h2>Class Ontology & Prompts</h2>
            <p style="color: #aaa; font-size: 14px;">Define YOLO classes. Check <b>Invert?</b> to extract the background around the subject.</p>
            
            <div style="background: var(--panel); border: 1px solid var(--border); border-radius: 6px; overflow: hidden;">
                <table>
                    <thead>
                        <tr>
                            <th style="width: 50px;">ID</th>
                            <th>Display Name</th>
                            <th>SAM 3 Text Prompt</th>
                            <th style="width: 70px; text-align:center;">Invert?</th>
                            <th style="width: 60px;">Color</th>
                            <th style="width: 50px;"></th>
                        </tr>
                    </thead>
                    <tbody id="ontologyTableBody"></tbody>
                </table>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 20px;">
                <button onclick="addNewClass()">+ Add New Class</button>
                <button onclick="saveConfig()" class="btn-success" style="padding: 10px 30px;">💾 Save Configuration</button>
            </div>
        </div>
    </main>

    <div id="toast"></div>

    <script>
        const AppState = {
            config: { input_dir: '', output_dir: '', classes: [] },
            gallery: [],
            galleryFilter: 'pending', 
            currentPath: '',
            selectedPaths: new Set(),
            queue: [],
            queueIndex: 0,
            imageObj: null,
            polygons: [],
            selectedPolyIndex: -1
        };

        const HEADERS = { 'ngrok-skip-browser-warning': 'true' };
        const HEADERS_JSON = { 'Content-Type': 'application/json', ...HEADERS };

        function showToast(msg, type = "success") {
            const toast = document.getElementById('toast');
            toast.textContent = msg;
            toast.style.borderLeftColor = type === 'error' ? 'var(--danger)' : 'var(--success)';
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }

        function switchView(viewName) {
            document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(`view-${viewName}`).classList.add('active');
            event.currentTarget.classList.add('active');
            
            if(viewName === 'data') renderGallery();
            if(viewName === 'ontology') renderOntology();
            if(viewName === 'studio') renderStudioClasses();
        }

        async function initApp() {
            try {
                const res = await fetch('/api/config', { headers: HEADERS });
                AppState.config = await res.json();
                renderStudioClasses();
                await refreshGallery(false);
            } catch (e) {
                showToast("Failed to connect to backend.", "error");
            }
        }

        // --- Data & Directory Explorer Logic ---
        async function refreshGallery(showMsg = true) {
            try {
                const res = await fetch('/api/gallery', { headers: HEADERS });
                const data = await res.json();
                AppState.gallery = data.images;
                renderGallery();
                if(showMsg) showToast("Directory refreshed");
            } catch (e) {
                showToast("Failed to load directory", "error");
            }
        }

        function setGalleryFilter(filter) {
            AppState.galleryFilter = filter;
            document.querySelectorAll('.sub-tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(`tab-${filter}`).classList.add('active');
            AppState.selectedPaths.clear(); 
            updateBulkActionBar();
            renderGallery();
        }

        function navigatePath(newPath) {
            AppState.currentPath = newPath;
            AppState.selectedPaths.clear();
            updateBulkActionBar();
            renderGallery();
        }

        function renderBreadcrumb() {
            const bc = document.getElementById('breadcrumbNav');
            bc.innerHTML = `<span onclick="navigatePath('')">🏠 Root</span>`;
            if(!AppState.currentPath) return;
            
            let parts = AppState.currentPath.split('/').filter(p => p);
            let accum = '';
            parts.forEach(p => {
                accum += p + '/';
                const curPath = accum;
                bc.innerHTML += ` <span style="color:#666; cursor:default;">/</span> <span onclick="navigatePath('${curPath}')">${p}</span>`;
            });
        }

        function renderGallery() {
            renderBreadcrumb();
            const container = document.getElementById('galleryContainer');
            container.innerHTML = '';
            
            let filesInPath = AppState.gallery.filter(img => img.filename.startsWith(AppState.currentPath));
            let folders = new Map();
            let items = [];

            filesInPath.forEach(img => {
                const relPath = img.filename.substring(AppState.currentPath.length);
                if (relPath.includes('/')) {
                    const folderName = relPath.split('/')[0];
                    if(!folders.has(folderName)) folders.set(folderName, []);
                    folders.get(folderName).push(img);
                } else {
                    items.push(img);
                }
            });

            // Filter logic
            const filter = AppState.galleryFilter;
            
            // Process Folders
            Array.from(folders.entries()).forEach(([name, contents]) => {
                const isApproved = contents.every(f => f.status === 'approved');
                if(filter === 'pending' && isApproved) return;
                if(filter === 'approved' && !isApproved) return;
                
                const fullFolderPath = AppState.currentPath + name + '/';
                const isChecked = AppState.selectedPaths.has(fullFolderPath) ? 'checked' : '';
                
                const div = document.createElement('div');
                div.className = 'compact-item folder';
                div.innerHTML = `
                    <div class="compact-left">
                        <input type="checkbox" ${isChecked} onclick="event.stopPropagation()" onchange="toggleSelection('${fullFolderPath}', this.checked)" style="cursor:pointer;">
                        <span style="font-size:20px;">📁</span>
                        <span><b>${name}</b> <span style="color:#888; font-size:12px;">(${contents.length} items)</span></span>
                    </div>
                    <span class="badge ${isApproved ? 'approved' : 'pending'}">${isApproved ? 'APPROVED' : 'PENDING'}</span>
                `;
                div.onclick = (e) => { if(e.target.tagName !== 'INPUT') navigatePath(fullFolderPath); };
                container.appendChild(div);
            });

            // Process Files
            items.forEach(img => {
                const isApproved = img.status === 'approved';
                if(filter === 'pending' && isApproved) return;
                if(filter === 'approved' && !isApproved) return;

                const isChecked = AppState.selectedPaths.has(img.filename) ? 'checked' : '';
                const baseName = img.filename.substring(AppState.currentPath.length);
                
                const div = document.createElement('div');
                div.className = 'compact-item';
                div.innerHTML = `
                    <div class="compact-left">
                        <input type="checkbox" ${isChecked} onchange="toggleSelection('${img.filename}', this.checked)" style="cursor:pointer;">
                        <span style="font-size:20px;">🖼️</span>
                        <span>${baseName}</span>
                    </div>
                    <span class="badge ${isApproved ? 'approved' : 'pending'}">${isApproved ? 'APPROVED' : 'PENDING'}</span>
                `;
                container.appendChild(div);
            });
            
            if(container.innerHTML === '') {
                container.innerHTML = `<div style="padding: 20px; text-align:center; color:#666;">No items found in this view.</div>`;
            }
        }

        function toggleSelection(path, isChecked) {
            if(isChecked) AppState.selectedPaths.add(path);
            else AppState.selectedPaths.delete(path);
            updateBulkActionBar();
        }

        function updateBulkActionBar() {
            const bar = document.getElementById('bulkActionBar');
            if(AppState.selectedPaths.size > 0) {
                document.getElementById('bulkCount').innerText = AppState.selectedPaths.size;
                bar.style.display = 'flex';
            } else {
                bar.style.display = 'none';
            }
        }

        function getFilesFromSelections() {
            let filesToProcess = new Set();
            AppState.selectedPaths.forEach(path => {
                if(path.endsWith('/')) {
                    // It's a folder, get all nested files
                    AppState.gallery.forEach(img => {
                        if(img.filename.startsWith(path)) filesToProcess.add(img.filename);
                    });
                } else {
                    filesToProcess.add(path);
                }
            });
            return Array.from(filesToProcess);
        }

        function loadSelectedIntoStudio() {
            const files = getFilesFromSelections();
            if(files.length === 0) return;
            AppState.queue = files;
            AppState.queueIndex = 0;
            switchView('studio');
            loadQueueItem();
            showToast(`Loaded ${files.length} items into Studio Queue`);
        }

        async function bulkDelete() {
            const files = getFilesFromSelections();
            if(!confirm(`Delete ${files.length} files permanently?`)) return;

            try {
                await Promise.all(files.map(f => fetch(`/api/image/${encodeURIComponent(f)}`, { method: 'DELETE', headers: HEADERS })));
                showToast(`Deleted ${files.length} files`);
                AppState.selectedPaths.clear();
                updateBulkActionBar();
                refreshGallery(false);
            } catch (e) {
                showToast("Deletions failed", "error");
            }
        }

        // --- Upload Logic ---
        async function handleUpload(e) {
            const files = e.target.files;
            if(files.length === 0) return;

            const formData = new FormData();
            for(let i=0; i<files.length; i++) {
                const path = files[i].webkitRelativePath || files[i].name;
                formData.append('images', files[i], path);
            }

            showToast(`Uploading ${files.length} files...`);
            try {
                const res = await fetch('/api/upload', { method: 'POST', headers: HEADERS, body: formData });
                const data = await res.json();
                if(data.success) { showToast(`Uploaded ${data.uploaded} files!`); refreshGallery(false); }
            } catch(err) { showToast(err.message, "error"); }
            e.target.value = "";
        }

        async function handleZipUpload(e) {
            const file = e.target.files[0];
            if(!file) return;
            const formData = new FormData();
            formData.append('zip', file);
            showToast(`Extracting ZIP on server...`);
            try {
                const res = await fetch('/api/upload_zip', { method: 'POST', headers: HEADERS, body: formData });
                const data = await res.json();
                if(data.success) { showToast(`Extracted ${data.uploaded} files!`); refreshGallery(false); }
            } catch(err) { showToast(err.message, "error"); }
            e.target.value = "";
        }

        // --- Ontology Logic ---
        function renderOntology() {
            const tbody = document.getElementById('ontologyTableBody');
            tbody.innerHTML = '';
            AppState.config.classes.forEach((cls, index) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><input type="text" id="cls-id-${index}" value="${cls.id}" style="text-align:center;"></td>
                    <td><input type="text" id="cls-name-${index}" value="${cls.name}"></td>
                    <td><input type="text" id="cls-prompt-${index}" value="${cls.prompt}"></td>
                    <td style="text-align:center;"><input type="checkbox" id="cls-invert-${index}" ${cls.invert ? 'checked' : ''} style="width:20px; height:20px; cursor:pointer;"></td>
                    <td><input type="color" id="cls-color-${index}" value="${cls.color}" style="height:35px; padding:2px;"></td>
                    <td><button onclick="removeOntologyRow(${index})" class="btn-danger" style="padding:8px;">🗑️</button></td>
                `;
                tbody.appendChild(tr);
            });
        }
        function addNewClass() {
            const nextId = AppState.config.classes.length > 0 ? Math.max(...AppState.config.classes.map(c => c.id)) + 1 : 0;
            AppState.config.classes.push({ id: nextId, name: "New Class", prompt: "describe it", invert: false, color: "#ffffff" });
            renderOntology();
        }
        function removeOntologyRow(index) { AppState.config.classes.splice(index, 1); renderOntology(); }
        async function saveConfig() {
            const newClasses = [];
            for(let i=0; i<AppState.config.classes.length; i++) {
                newClasses.push({
                    id: parseInt(document.getElementById(`cls-id-${i}`).value),
                    name: document.getElementById(`cls-name-${i}`).value,
                    prompt: document.getElementById(`cls-prompt-${i}`).value,
                    invert: document.getElementById(`cls-invert-${i}`).checked,
                    color: document.getElementById(`cls-color-${i}`).value
                });
            }
            AppState.config.classes = newClasses;
            await fetch('/api/config', { method: 'POST', headers: HEADERS_JSON, body: JSON.stringify(AppState.config) });
            showToast("Config saved!");
            renderStudioClasses();
        }

        // --- Studio & Queue Logic ---
        const canvas = document.getElementById('editorCanvas');
        const ctx = canvas.getContext('2d');

        function renderStudioClasses() {
            const container = document.getElementById('activeClassesList');
            container.innerHTML = '';
            AppState.config.classes.forEach(cls => {
                const row = document.createElement('div');
                row.className = `class-row ${AppState.selectedPolyIndex !== -1 && AppState.polygons[AppState.selectedPolyIndex]?.classId === cls.id ? 'selected' : ''}`;
                const cb = document.createElement('input');
                cb.type = 'checkbox'; cb.className = 'sam-target-cb'; cb.value = cls.id; cb.checked = true;
                const btn = document.createElement('button');
                btn.className = 'class-btn';
                btn.innerHTML = `<span style="display:flex; align-items:center; gap:8px;"><div class="color-box" style="background-color: ${cls.color}"></div> ${cls.name} ${cls.invert ? '<span style="color:#f0ad4e; font-size:10px;">(Inv)</span>' : ''}</span><span style="opacity:0.5; font-size:10px;">ID:${cls.id}</span>`;
                btn.onclick = () => assignClassToSelected(cls.id);
                row.appendChild(cb); row.appendChild(btn); container.appendChild(row);
            });
        }
        function toggleAllSAM(checked) { document.querySelectorAll('.sam-target-cb').forEach(cb => cb.checked = checked); }
        function assignClassToSelected(cid) { if (AppState.selectedPolyIndex !== -1) { AppState.polygons[AppState.selectedPolyIndex].classId = cid; renderStudioClasses(); drawCanvas(); } }

        function navQueue(dir) {
            if(AppState.queue.length === 0) return;
            AppState.queueIndex += dir;
            if(AppState.queueIndex < 0) AppState.queueIndex = 0;
            if(AppState.queueIndex >= AppState.queue.length) {
                AppState.queueIndex = AppState.queue.length - 1;
                showToast("End of queue reached.");
                return;
            }
            loadQueueItem();
        }

        async function loadQueueItem() {
            if(AppState.queue.length === 0) return;
            const filename = AppState.queue[AppState.queueIndex];
            
            document.getElementById('queueCount').innerText = `${AppState.queueIndex + 1} / ${AppState.queue.length}`;
            document.getElementById('currentFilename').innerText = "Loading...";
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            try {
                const res = await fetch(`/api/image/${encodeURIComponent(filename)}/data`, { headers: HEADERS });
                const data = await res.json();
                if (!data.success) throw new Error(data.error);

                document.getElementById('currentFilename').innerText = filename;
                AppState.imageObj = new Image();
                AppState.imageObj.onload = () => {
                    canvas.width = AppState.imageObj.width;
                    canvas.height = AppState.imageObj.height;
                    AppState.polygons = data.annotations || [];
                    AppState.selectedPolyIndex = -1;
                    renderStudioClasses();
                    drawCanvas();
                };
                AppState.imageObj.src = "data:image/jpeg;base64," + data.image_b64;
            } catch(e) { showToast(e.message, "error"); }
        }

        function getActivePrompts() {
            const activeIds = Array.from(document.querySelectorAll('.sam-target-cb:checked')).map(cb => parseInt(cb.value));
            return AppState.config.classes.filter(c => activeIds.includes(c.id));
        }

        async function runSAM3Single() {
            if (AppState.queue.length === 0) return;
            const prompts = getActivePrompts();
            if(prompts.length === 0) return showToast("Select at least one class checkbox", "error");
            
            document.getElementById('samLoader').style.display = 'flex';
            try {
                const fname = AppState.queue[AppState.queueIndex];
                const res = await fetch('/api/auto_label', {
                    method: 'POST', headers: HEADERS_JSON,
                    body: JSON.stringify({ filename: fname, prompts: prompts })
                });
                const data = await res.json();
                if (data.success) { AppState.polygons.push(...data.polygons); drawCanvas(); }
            } catch(e) { showToast("SAM Error", "error"); } 
            finally { document.getElementById('samLoader').style.display = 'none'; }
        }

        async function startBulkRun() {
            if(AppState.queue.length === 0) return;
            const prompts = getActivePrompts();
            if(prompts.length === 0) return showToast("Select at least one class checkbox", "error");
            if(!confirm(`Run Auto-Labeler on ${AppState.queue.length} items? This will save directly to disk.`)) return;

            const loader = document.getElementById('bulkLoader');
            const fill = document.getElementById('bulkProgressFill');
            const statText = document.getElementById('bulkStatusText');
            const etaText = document.getElementById('bulkEtaText');
            
            loader.style.display = 'flex';
            fill.style.width = '0%';
            
            try {
                const res = await fetch('/api/auto_label_bulk', {
                    method: 'POST', headers: HEADERS_JSON,
                    body: JSON.stringify({ filenames: AppState.queue, prompts: prompts })
                });

                const reader = res.body.getReader();
                const decoder = new TextDecoder("utf-8");
                let startTime = Date.now();
                let total = AppState.queue.length;

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    
                    const chunks = decoder.decode(value).split('\n\n');
                    for(let chunk of chunks) {
                        if(!chunk.startsWith('data: ')) continue;
                        const data = JSON.parse(chunk.replace('data: ', ''));
                        
                        if(data.status === 'progress') {
                            const pct = (data.current / total) * 100;
                            fill.style.width = pct + '%';
                            statText.innerText = `Processed ${data.current} / ${total} \n (${data.filename})`;
                            
                            const elapsed = (Date.now() - startTime) / 1000;
                            const avg = elapsed / data.current;
                            const remain = Math.round(avg * (total - data.current));
                            
                            const mins = Math.floor(remain / 60);
                            const secs = remain % 60;
                            etaText.innerText = `ETA: ${mins}:${secs.toString().padStart(2, '0')}`;
                        } else if (data.status === 'error') {
                            console.error("Error on file:", data.filename, data.error);
                        }
                    }
                }
                showToast("Bulk processing complete!");
                refreshGallery(false);
                loadQueueItem(); // Reload current view to show new polygons
            } catch(e) {
                showToast("Stream disconnected.", "error");
            } finally {
                loader.style.display = 'none';
            }
        }

        async function saveAndNext() {
            if (AppState.queue.length === 0) return;
            const fname = AppState.queue[AppState.queueIndex];
            try {
                const res = await fetch('/api/save', {
                    method: 'POST', headers: HEADERS_JSON,
                    body: JSON.stringify({ filename: fname, annotations: AppState.polygons })
                });
                const data = await res.json();
                if (data.success) {
                    const img = AppState.gallery.find(i => i.filename === fname);
                    if(img) img.status = 'approved';
                    navQueue(1);
                }
            } catch(e) {}
        }

        function clearPolygons() { AppState.polygons = []; AppState.selectedPolyIndex = -1; renderStudioClasses(); drawCanvas(); }

        // Canvas Interaction
        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width; const scaleY = canvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX / canvas.width; const y = (e.clientY - rect.top) * scaleY / canvas.height;
            AppState.selectedPolyIndex = -1;
            for(let i = AppState.polygons.length - 1; i >= 0; i--) {
                if(pointInPolygon({x, y}, AppState.polygons[i].points)) { AppState.selectedPolyIndex = i; break; }
            }
            renderStudioClasses(); drawCanvas();
        });

        window.addEventListener('keydown', (e) => {
            if ((e.key === 'Delete' || e.key === 'Backspace') && AppState.selectedPolyIndex !== -1) {
                AppState.polygons.splice(AppState.selectedPolyIndex, 1);
                AppState.selectedPolyIndex = -1; renderStudioClasses(); drawCanvas();
            }
        });

        function pointInPolygon(point, vs) {
            let x = point.x, y = point.y, inside = false;
            for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
                let xi = vs[i].x, yi = vs[i].y, xj = vs[j].x, yj = vs[j].y;
                let intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
                if (intersect) inside = !inside;
            }
            return inside;
        }

        function drawCanvas() {
            if (!AppState.imageObj) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(AppState.imageObj, 0, 0, canvas.width, canvas.height);

            AppState.polygons.forEach((poly, idx) => {
                ctx.beginPath();
                poly.points.forEach((p, i) => {
                    const px = p.x * canvas.width; const py = p.y * canvas.height;
                    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
                });
                ctx.closePath();

                const cls = AppState.config.classes.find(c => c.id === poly.classId);
                const color = cls ? cls.color : '#ffffff';
                
                ctx.fillStyle = color + '66'; ctx.fill();
                ctx.lineWidth = idx === AppState.selectedPolyIndex ? 4 : 2;
                ctx.strokeStyle = idx === AppState.selectedPolyIndex ? '#ffffff' : color;
                if (idx === AppState.selectedPolyIndex) ctx.setLineDash([5, 5]); else ctx.setLineDash([]);
                ctx.stroke(); ctx.setLineDash([]);
            });
        }

        window.onload = initApp;
    </script>
</body>
</html>
```

---

### Cell 4: Create the Streaming Flask Backend (`app.py`)

This cell supports recursive directories, path sanitization, streaming chunked responses, and forceful VRAM garbage collection.

```python
%%writefile app.py
import os
import json
import shutil
import base64
import torch
import cv2
import zipfile
import gc
import urllib.parse
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

print("--> Loading SAM 3 Model (This takes a minute)...", flush=True)

BPE_PATH = "/content/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(bpe_path=BPE_PATH).cuda().eval()

def make_fp32_safe(m):
    for name, param in m.named_parameters(recurse=False):
        if param.dtype in [torch.bfloat16, torch.float16]: param.data = param.data.to(torch.float32)
    for name, buf in m.named_buffers(recurse=False):
        if buf.dtype in [torch.bfloat16, torch.float16]: buf.data = buf.data.to(torch.float32)

for m in model.modules(): make_fp32_safe(m)

processor = Sam3Processor(model)
if hasattr(processor, 'autocast'): processor.autocast = torch.autocast(device_type="cuda", dtype=torch.float32)

print("--> Model loaded successfully!", flush=True)

CONFIG_FILE = "/content/drive/MyDrive/SAM3_YOLO/config.json"
DEFAULT_CONFIG = {
    "input_dir": "/content/drive/MyDrive/SAM3_YOLO/raw_images",
    "output_dir": "/content/drive/MyDrive/SAM3_YOLO/dataset",
    "classes": [
        {"id": 0, "name": "Foreground", "prompt": "the main subject", "invert": False, "color": "#00ff00"}
    ]
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f: return json.load(f)
    return DEFAULT_CONFIG

def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w') as f: json.dump(cfg, f, indent=4)

def secure_rel_path(path):
    # Keep slashes, but prevent directory traversal exploits
    clean = os.path.normpath(path).replace('..', '')
    return clean.lstrip('/\\')

def mask_to_yolo_polygons(binary_mask, invert=False):
    binary_mask = np.squeeze(binary_mask)
    if binary_mask.ndim != 2: return []
    if invert: binary_mask = np.logical_not(binary_mask)
    h, w = binary_mask.shape
    mask_uint8 = np.ascontiguousarray((binary_mask * 255).astype(np.uint8))
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3: continue
        polygons.append([{"x": float(pt[0][0]) / w, "y": float(pt[0][1]) / h} for pt in approx])
    return polygons

def parse_yolo_txt(txt_path):
    polygons = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7: 
                    cls_id = int(parts[0])
                    pts = [{"x": float(parts[i]), "y": float(parts[i+1])} for i in range(1, len(parts), 2)]
                    polygons.append({"classId": cls_id, "points": pts})
    return polygons

# --- API Routes ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'GET': return jsonify(load_config())
    save_config(request.json); return jsonify({"success": True})

@app.route('/api/gallery', methods=['GET'])
def api_gallery():
    cfg = load_config()
    images = []
    input_dir = cfg['input_dir']
    if not os.path.exists(input_dir): os.makedirs(input_dir)
    
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, input_dir).replace('\\', '/')
                txt_path = os.path.join(cfg['output_dir'], "labels", os.path.splitext(rel_path)[0] + ".txt")
                status = "approved" if os.path.exists(txt_path) else "pending"
                images.append({"filename": rel_path, "status": status})
    
    # Sort files naturally
    images = sorted(images, key=lambda x: x['filename'])
    return jsonify({"success": True, "images": images})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    cfg = load_config()
    count = 0
    for file in request.files.getlist('images'):
        if file.filename == '': continue
        safe_path = secure_rel_path(file.filename)
        dest = os.path.join(cfg['input_dir'], safe_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        file.save(dest)
        count += 1
    return jsonify({"success": True, "uploaded": count})

@app.route('/api/upload_zip', methods=['POST'])
def api_upload_zip():
    cfg = load_config()
    zip_file = request.files.get('zip')
    if not zip_file: return jsonify({"success": False})
    
    tmp_path = os.path.join(cfg['input_dir'], "temp_upload.zip")
    zip_file.save(tmp_path)
    count = 0
    with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            if member.lower().endswith(('.png', '.jpg', '.jpeg')) and not member.startswith('__MACOSX'):
                safe_name = secure_rel_path(member)
                dest = os.path.join(cfg['input_dir'], safe_name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with zip_ref.open(member) as src, open(dest, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                count += 1
    os.remove(tmp_path)
    return jsonify({"success": True, "uploaded": count})

@app.route('/api/image/<path:filename>', methods=['DELETE'])
def api_delete(filename):
    cfg = load_config()
    fname = urllib.parse.unquote(filename)
    
    paths_to_delete = [
        os.path.join(cfg['input_dir'], fname),
        os.path.join(cfg['output_dir'], "labels", os.path.splitext(fname)[0] + ".txt"),
        os.path.join(cfg['output_dir'], "images", fname)
    ]
    for p in paths_to_delete:
        if os.path.exists(p): os.remove(p)
    return jsonify({"success": True})

@app.route('/api/image/<path:filename>/data', methods=['GET'])
def api_image_data(filename):
    cfg = load_config()
    fname = urllib.parse.unquote(filename)
    img_path = os.path.join(cfg['input_dir'], fname)
    txt_path = os.path.join(cfg['output_dir'], "labels", os.path.splitext(fname)[0] + ".txt")
    
    if not os.path.exists(img_path): return jsonify({"success": False, "error": "Not found"})
    with open(img_path, "rb") as f: b64_string = base64.b64encode(f.read()).decode('utf-8')
    return jsonify({"success": True, "image_b64": b64_string, "annotations": parse_yolo_txt(txt_path)})

# Base logic for single inference
def infer_image(img_path, prompts):
    image = Image.open(img_path).convert("RGB")
    results = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float32):
        inference_state = processor.set_image(image)
        for cls in prompts:
            output = processor.set_text_prompt(state=inference_state, prompt=cls['prompt'])
            masks = output["masks"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            for i, mask in enumerate(masks):
                if scores[i] < 0.50: continue
                polys = mask_to_yolo_polygons(mask, invert=cls.get('invert', False))
                for p in polys: results.append({"classId": cls['id'], "points": p})
    del image; del inference_state
    return results

@app.route('/api/auto_label', methods=['POST'])
def api_auto_label():
    data = request.json
    try:
        results = infer_image(os.path.join(load_config()['input_dir'], data['filename']), data['prompts'])
        torch.cuda.empty_cache(); gc.collect()
        return jsonify({"success": True, "polygons": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# SSE Streaming Endpoint for bulk processing
@app.route('/api/auto_label_bulk', methods=['POST'])
def api_auto_label_bulk():
    data = request.json
    files = data['filenames']
    prompts = data['prompts']
    cfg = load_config()

    def generate():
        yield f"data: {json.dumps({'status': 'start', 'total': len(files)})}\n\n"
        for i, fname in enumerate(files):
            try:
                results = infer_image(os.path.join(cfg['input_dir'], fname), prompts)
                
                # Save to specific subdirectories
                out_lbl_dir = os.path.join(cfg['output_dir'], "labels", os.path.dirname(fname))
                out_img_dir = os.path.join(cfg['output_dir'], "images", os.path.dirname(fname))
                os.makedirs(out_lbl_dir, exist_ok=True)
                os.makedirs(out_img_dir, exist_ok=True)

                dst_img = os.path.join(cfg['output_dir'], "images", fname)
                src_img = os.path.join(cfg['input_dir'], fname)
                if not os.path.exists(dst_img) and os.path.exists(src_img): shutil.copy(src_img, dst_img)

                txt_path = os.path.join(cfg['output_dir'], "labels", os.path.splitext(fname)[0] + ".txt")
                with open(txt_path, 'w') as f:
                    for ann in results:
                        pstr = " ".join([f"{pt['x']:.6f} {pt['y']:.6f}" for pt in ann['points']])
                        f.write(f"{ann['classId']} {pstr}\n")

                yield f"data: {json.dumps({'status': 'progress', 'current': i+1, 'filename': fname})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'status': 'error', 'filename': fname, 'error': str(e)})}\n\n"
            finally:
                torch.cuda.empty_cache()
                gc.collect() # Extremely important to prevent T4 OOM in bulk runs
        
        yield f"data: {json.dumps({'status': 'done'})}\n\n"

    return app.response_class(generate(), mimetype='text/event-stream')

@app.route('/api/save', methods=['POST'])
def api_save():
    data = request.json
    fname = data['filename']
    cfg = load_config()
    
    out_lbl_dir = os.path.join(cfg['output_dir'], "labels", os.path.dirname(fname))
    out_img_dir = os.path.join(cfg['output_dir'], "images", os.path.dirname(fname))
    os.makedirs(out_lbl_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)

    try:
        src_img = os.path.join(cfg['input_dir'], fname)
        dst_img = os.path.join(cfg['output_dir'], "images", fname)
        if not os.path.exists(dst_img) and os.path.exists(src_img): shutil.copy(src_img, dst_img)

        txt_path = os.path.join(cfg['output_dir'], "labels", os.path.splitext(fname)[0] + ".txt")
        with open(txt_path, 'w') as f:
            for ann in data['annotations']:
                pstr = " ".join([f"{pt['x']:.6f} {pt['y']:.6f}" for pt in ann['points']])
                f.write(f"{ann['classId']} {pstr}\n")
        return jsonify({"success": True})
    except Exception as e: return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

### Cell 5: Start the Server & Open the UI
This cell authenticates Ngrok, kills old stray processes, starts Flask in the background, and provides your secure public URL.

```python
import subprocess
import time
import getpass
import os
from pyngrok import ngrok
from google.colab import userdata

# 1. Authenticate with Ngrok and Hugging Face
try:
    ngrok_token = userdata.get('NGROK_TOKEN')
    os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
except Exception:
    print("Colab secrets not found. Please paste manually:")
    ngrok_token = getpass.getpass("Ngrok Authtoken: ")
    os.environ["HF_TOKEN"] = getpass.getpass("Hugging Face Token: ")

ngrok.set_auth_token(ngrok_token.strip())

# 2. Kill old processes to prevent port blocking
os.system("pkill -f -9 'app.py'")
ngrok.kill()
time.sleep(1)

# 3. Start Flask App in the background
print("Starting Flask Server... (Logs piping to flask_logs.txt)")
log_file = open("flask_logs.txt", "w")
flask_process = subprocess.Popen(["python", "-u", "app.py"], stdout=log_file, stderr=subprocess.STDOUT)

# 4. Wait for server to boot (SAM 3 takes about 30 seconds to load into VRAM)
print("Loading SAM 3 model weights... Please wait.")
time.sleep(20) # Conservative wait

# 5. Open Ngrok Tunnel
public_url = ngrok.connect(addr="127.0.0.1:5000").public_url

print("="*60)
print(f"✅ READY! Click the link below to open the Annotation UI:")
print(f"👉 {public_url}")
print("="*60)
print("Note: If the page doesn't load immediately, SAM 3 is still booting. Wait 15 seconds and refresh.")
```