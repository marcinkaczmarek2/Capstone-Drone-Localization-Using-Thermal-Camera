%% ============== KONFIGURACJA ==============
inputFolder = "C:\Users\szymo\DroneLocalization\dataset\Drone-detection-dataset\Data\Video_IR";
outputFolder = "C:\Users\szymo\DroneLocalization\dataset\Drone-detection-dataset\Data\Video_IR_output";

% Mapowanie klas - case-insensitive
classMap = containers.Map({'DRONE','AIRPLANE','BIRD','HELICOPTER'},{0,1,1,1});

% FPS fallback (jeśli VideoReader.FrameRate nie jest dostępny)
FPS_FALLBACK = 30;

% Pobierz wszystkie pliki mp4 w folderze
videoFiles = dir(fullfile(inputFolder,'*.mp4'));

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

%% ============== PĘTLA GŁÓWNA ==============
for v = 1:length(videoFiles)
    [~, baseName, ~] = fileparts(videoFiles(v).name);
    fprintf("\n--- Processing %s ---\n", baseName);
    try
        mp4Path = fullfile(inputFolder, videoFiles(v).name);
        matPath = fullfile(inputFolder, baseName + "_LABELS.mat");
        
        if ~isfile(matPath)
            fprintf("Brak MAT dla %s – pomijam\n", baseName);
            continue;
        end
        
        % load groundTruth
        data = load(matPath);
        vars = fieldnames(data);
        g = data.(vars{1}); % groundTruth object
        
        % LabelData (tabela)
        labelData = g.LabelData;
        % labelDefs przyjmujemy nazwy kolumn (np. 'AIRPLANE','BIRD','DRONE','HELICOPTER')
        try
            labelDefs = g.LabelDefinitions.Name;
        catch
            % fallback: nazwy kolumn tabeli (jeśli LabelDefinitions nie istnieje)
            labelDefs = labelData.Properties.VariableNames;
        end
        
        % przygotuj mapę klas w trybie case-insensitive
        % upewnij się, że wszystkie nazwy mapują się do wartości
        for k = 1:length(labelDefs)
            lbl = upper(string(labelDefs{k}));
            if ~isKey(classMap,lbl)
                % jeśli nowa klasa pojawiła się, mapuj ją do notdrone = 1
                classMap(lbl) = 1;
            end
        end
        
        % VideoReader
        vidObj = VideoReader(mp4Path);
        imgW = vidObj.Width;
        imgH = vidObj.Height;
        if isprop(vidObj,'FrameRate')
            fps = vidObj.FrameRate;
            if isempty(fps) || fps==0
                fps = FPS_FALLBACK;
            end
        else
            fps = FPS_FALLBACK;
        end
        
        % Budujemy listę docelowych indeksów klatek i bboxów
        % keys: frameIndex -> cell array of lines (strings) to write to txt
        frameBoxes = containers.Map('KeyType','double','ValueType','any');
        
        nRows = height(labelData);
        if nRows == 0
            fprintf("Empty LabelData for %s — skipping\n", baseName);
            continue;
        end
        
        % detect how to read time:
        hasTime = ismember('Time', labelData.Properties.VariableNames);
        
        for i = 1:nRows
            % compute time in seconds
            if hasTime
                tVal = labelData.Time(i);
                % handle duration or numeric
                if isa(tVal,'duration')
                    tsec = seconds(tVal);
                elseif isnumeric(tVal)
                    tsec = double(tVal);
                else
                    % try convert
                    try
                        tsec = seconds(tVal);
                    catch
                        tsec = double(tVal);
                    end
                end
                frameIdx = max(1, round(tsec * fps)); % frame index (1-based)
            else
                % No Time column: assume consecutive frames starting at frame 1
                frameIdx = i; % assume 1->frame1, etc.
            end
            
            % for each class column, extract bbox(s)
            for c = 1:length(labelDefs)
                className = labelDefs{c};
                try
                    cellVal = labelData.(className){i}; % could be [] or Nx4
                catch
                    cellVal = [];
                end
                if isempty(cellVal)
                    continue;
                end
                % cellVal might be Mx4 numeric array or 1x4 row
                if isnumeric(cellVal)
                    % if 2D numeric, iterate rows
                    if ismatrix(cellVal) && size(cellVal,2) == 4
                        for r = 1:size(cellVal,1)
                            bbox = cellVal(r,:);
                            % convert to YOLO normalized format
                            x = bbox(1); y = bbox(2); w = bbox(3); h = bbox(4);
                            x_center = (x + w/2) / imgW;
                            y_center = (y + h/2) / imgH;
                            w_norm = w / imgW;
                            h_norm = h / imgH;
                            % class id (case-insensitive)
                            cidKey = upper(string(className));
                            if isKey(classMap, cidKey)
                                cid = classMap(cidKey);
                            else
                                cid = 1; % default notdrone
                                classMap(cidKey) = cid;
                            end
                            line = sprintf('%d %.6f %.6f %.6f %.6f', cid, x_center, y_center, w_norm, h_norm);
                            if isKey(frameBoxes, frameIdx)
                                arr = frameBoxes(frameIdx);
                                arr{end+1} = line;
                                frameBoxes(frameIdx) = arr;
                            else
                                frameBoxes(frameIdx) = {line};
                            end
                        end
                    else
                        % unexpected format: try to flatten
                        try
                            bbox = double(cellVal(:))';
                            if numel(bbox) >= 4
                                bbox = bbox(1:4);
                                x = bbox(1); y = bbox(2); w = bbox(3); h = bbox(4);
                                x_center = (x + w/2) / imgW;
                                y_center = (y + h/2) / imgH;
                                w_norm = w / imgW;
                                h_norm = h / imgH;
                                cidKey = upper(string(className));
                                cid = isKey(classMap,cidKey) * classMap(cidKey) + (~isKey(classMap,cidKey))*1;
                                line = sprintf('%d %.6f %.6f %.6f %.6f', cid, x_center, y_center, w_norm, h_norm);
                                frameBoxes(frameIdx) = {line};
                            end
                        catch
                            % skip if cannot parse
                        end
                    end
                else
                    % not numeric - skip
                    continue;
                end
            end
        end % end rows loop
        
        % If no frames with boxes, warn and skip
        if isempty(keys(frameBoxes))
            fprintf("No labeled frames found for %s — skipping\n", baseName);
            continue;
        end
        
        % Create output dir
        outDir = fullfile(outputFolder, baseName);
        if ~exist(outDir,'dir')
            mkdir(outDir);
        end
        
        % Now read video sequentially and write frames that are in frameBoxes
        vidObj.CurrentTime = 0;
        frameCounter = 0;
        targetFrames = sort(cell2mat(keys(frameBoxes)));
        targetIdx = 1;
        nextTarget = targetFrames(targetIdx);
        
        % Read until we pass last needed frame or video ends
        while hasFrame(vidObj) && targetIdx <= length(targetFrames)
            frame = readFrame(vidObj);
            frameCounter = frameCounter + 1;
            if frameCounter == nextTarget
                % save image
                frameNumStr = sprintf('%06d', frameCounter);
                imgFile = fullfile(outDir, frameNumStr + ".jpg");
                imwrite(frame, imgFile);
                % save txt lines
                txtFile = fullfile(outDir, frameNumStr + ".txt");
                fid = fopen(txtFile,'w');
                lines = frameBoxes(nextTarget);
                for L = 1:length(lines)
                    fprintf(fid, "%s\n", lines{L});
                end
                fclose(fid);
                
                fprintf("Saved frame %d for %s (img+txt)\n", frameCounter, baseName);
                
                % advance target
                targetIdx = targetIdx + 1;
                if targetIdx <= length(targetFrames)
                    nextTarget = targetFrames(targetIdx);
                end
            end
            % if frameCounter > last target, break
        end
        
        fprintf("Done for %s: wrote %d labeled frames.\n", baseName, frameCounter >= targetFrames(end) * 1);
        
    catch ME
        fprintf("Błąd przy %s: %s\n", baseName, ME.message);
        continue;
    end
end

fprintf("\n===== ALL DONE =====\n");
