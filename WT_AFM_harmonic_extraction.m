function [amp_all, phase_f1, freq_labels] = WT_AFM_harmonic_extraction(sig, fd, fs, f1, num_harmonics, f2, options)
% WT_AFM_harmonic_extraction  Extract amplitude envelopes and phase for harmonic
%   components of a signal using Wavelet Packet Transform + CWT.
%
%   Combines a Wavelet Packet decomposition (to isolate frequency bands) with
%   a Continuous Wavelet Transform (to recover instantaneous amplitude) and
%   Wavelet Coherence (for phase relative to a reference/drive signal).
%
% -------------------------------------------------------------------------
%   SYNTAX
% -------------------------------------------------------------------------
%   [amp_all, phase_f1, freq_labels] = WT_AFM_harmonic_extraction(sig, fd, fs, f1, num_harmonics, f2)
%   [amp_all, phase_f1, freq_labels] = WT_AFM_harmonic_extraction(sig, fd, fs, f1, num_harmonics, f2, options)
%
% -------------------------------------------------------------------------
%   REQUIRED INPUTS
% -------------------------------------------------------------------------
%   sig           - (1 x N) or (N x 1)  Signal under analysis.
%   fd            - (1 x N) or (N x 1)  Drive / reference signal (same
%                   length as sig). Used to compute wavelet coherence phase.
%   fs            - Sampling frequency in Hz (positive scalar).
%   f1            - Fundamental frequency in Hz.  Harmonics are k*f1 for
%                   k = 1 … num_harmonics.
%   num_harmonics - Number of harmonics of f1 to process (positive integer).
%   f2            - Second frequency of interest in Hz.
%
% -------------------------------------------------------------------------
%   OPTIONAL NAME-VALUE INPUTS  (pass as  ..., 'Name', Value)
% -------------------------------------------------------------------------
%   'Wavelet'       Wavelet family string accepted by MODWPTDETAILS.
%                   Default: 'db45'
%   'WPTLevel'      Decomposition level for MODWPTDETAILS.
%                   Default: 7
%   'Padding'       Number of zero-padding samples added to each chunk to
%                   suppress edge artefacts.  Default: 2000
%   'ChunkSize'     Number of samples processed per loop iteration.
%                   Reduce if you hit memory limits.  Default: 1e6
%   'BandwidthExp'  Controls the CWT frequency-search bandwidth around each
%                   target frequency.  The search window is
%                   f_target * (exp(BandwidthExp)-1)/2.  Default: 0.1
%   'Verbose'       true/false  – print progress to the command window.
%                   Default: true
%
% -------------------------------------------------------------------------
%   OUTPUTS
% -------------------------------------------------------------------------
%   amp_all      - (num_harmonics+1 x N) Instantaneous amplitude matrix.
%                  Rows 1…num_harmonics correspond to harmonics f1, 2f1, …
%                  Row num_harmonics+1 corresponds to f2.
%   phase_f1     - (1 x N) Instantaneous phase (radians) of the wavelet
%                  coherence between sig and fd at frequency f1.
%   freq_labels  - Cell array of strings labelling each row of amp_all,
%                  e.g. {'f1 (29566 Hz)', '2f1 (59132 Hz)', ..., 'f2 (219345 Hz)'}
%
% -------------------------------------------------------------------------
%   EXAMPLE
% -------------------------------------------------------------------------
%   fs  = 1e6;
%   f1  = 29.566e3;
%   f2  = 219.345e3;
%   t   = (0:fs-1)/fs;
%   sig = sin(2*pi*f1*t) + 0.3*sin(2*pi*f2*t) + 0.05*randn(size(t));
%   fd  = sin(2*pi*f1*t + 0.4);          % drive signal with known phase
%
%   [amp, phi, labels] = WT_AFM_harmonic_extraction(sig, fd, fs, f1, 3, f2);
%
%   figure;
%   for k = 1:size(amp,1)
%       subplot(size(amp,1)+1, 1, k);
%       plot(t, amp(k,:));  ylabel(labels{k});
%   end
%   subplot(size(amp,1)+1, 1, size(amp,1)+1);
%   plot(t, phi);  ylabel('Phase f1 (rad)');  xlabel('Time (s)');
%
% -------------------------------------------------------------------------
%   DEPENDENCIES
% -------------------------------------------------------------------------
%   Requires MATLAB Wavelet Toolbox (modwptdetails, cwt, wcoherence).
%
% -------------------------------------------------------------------------
%   LICENSE  –  MIT  (see LICENSE in repository root)
% -------------------------------------------------------------------------
%   Author:   [Your Name]
%   Version:  1.0.0   (2026-05-25)
% -------------------------------------------------------------------------

    arguments
        sig           (1,:) {mustBeNumeric, mustBeFinite, mustBeNonempty, mustBeReal}
        fd            (1,:) {mustBeNumeric, mustBeFinite, mustBeNonempty, mustBeReal}
        fs            (1,1) {mustBePositive, mustBeFinite}
        f1            (1,1) {mustBePositive, mustBeFinite}
        num_harmonics (1,1) {mustBePositive, mustBeInteger}
        f2            (1,1) {mustBePositive, mustBeFinite}
        options.Wavelet       (1,:) char   = 'db45'
        options.WPTLevel      (1,1)        = 7
        options.Padding       (1,1) {mustBeNonnegative, mustBeInteger} = 2000
        options.ChunkSize     (1,1) {mustBePositive,    mustBeInteger} = 1e6
        options.BandwidthExp  (1,1) {mustBePositive}                   = 0.1
        options.Verbose       (1,1) logical                            = true
    end

    % ------------------------------------------------------------------ %
    %  0.  Validate inputs & issue warnings
    % ------------------------------------------------------------------ %
    sig = sig(:).';   % ensure row vector
    fd  = fd(:).';

    if length(sig) ~= length(fd)
        error('WT_AFM_harmonic_extraction:lengthMismatch', ...
              'sig and fd must have the same number of samples (got %d vs %d).', ...
              length(sig), length(fd));
    end

    nyquist = fs / 2;

    % --- Nyquist checks ---
    warn_ids = {};
    if f1 >= nyquist
        error('WT_AFM_harmonic_extraction:f1AboveNyquist', ...
              'f1 (%.3g Hz) is at or above the Nyquist frequency (%.3g Hz).', f1, nyquist);
    end

    if f2 >= nyquist
        error('WT_AFM_harmonic_extraction:f2AboveNyquist', ...
              'f2 (%.3g Hz) is at or above the Nyquist frequency (%.3g Hz).', f2, nyquist);
    end

    % Warn about harmonics that exceed Nyquist
    max_safe_harmonic = floor(nyquist / f1);
    if num_harmonics > max_safe_harmonic
        warning('WT_AFM_harmonic_extraction:harmonicsAboveNyquist', ...
                ['Requested %d harmonics, but harmonic %d (%.3g Hz) and above exceed ' ...
                 'the Nyquist frequency (%.3g Hz). Only harmonics 1…%d will be computed.'], ...
                num_harmonics, max_safe_harmonic+1, (max_safe_harmonic+1)*f1, nyquist, max_safe_harmonic);
        num_harmonics = max_safe_harmonic;
    end

    % Warn if f1 and f2 are very close
    if abs(f2 - f1) < 0.01 * f1
         warning('WT_AFM_harmonic_extraction:f1f2TooClose', ...
        ['f1 (%.3g Hz) and f2 (%.3g Hz) are within 1%% of each other. ' ...
         'Their WPT bands may overlap, which could reduce accuracy.'], ...
        f1, f2);
    end

   

    % Warn if any harmonic is close to f2
    harmonic_freqs = (1:num_harmonics) * f1;
    for k = 1:num_harmonics
        if abs(harmonic_freqs(k) - f2) < 0.01 * f2
            warning('WT_AFM_harmonic_extraction:harmonicNearF2', ...
            ['Harmonic %d (%.3g Hz) is within 1%% of f2 (%.3g Hz). ' ...
            'Results for this harmonic may be unreliable.'], ...
             k, harmonic_freqs(k), f2);
        end
    end

    % Short-signal warning
    min_recommended = 10 * options.ChunkSize;
    if length(sig) < options.ChunkSize
        warning('WT_AFM_harmonic_extraction:shortSignal', ...
            ['Signal length (%d samples) is shorter than one chunk (%d samples). ' ...
            'Edge padding effects may be significant.'], ...
            length(sig), options.ChunkSize);
    end

    % Toolbox check
    if ~license('test', 'Wavelet_Toolbox')
        error('WT_AFM_harmonic_extraction:noWaveletToolbox', ...
              'MATLAB Wavelet Toolbox is required but not available on this licence.');
    end

    % ------------------------------------------------------------------ %
    %  1.  Build frequency labels
    % ------------------------------------------------------------------ %
    freq_labels = cell(1, num_harmonics + 1);
    for k = 1:num_harmonics
        freq_labels{k} = sprintf('%df1 (%.4g Hz)', k, k*f1);
    end
    freq_labels{end} = sprintf('f2 (%.4g Hz)', f2);

    if options.Verbose
        fprintf('\n=== WT_AFM_harmonic_extraction ===\n');
        fprintf('  Signal length : %d samples  (%.3g s)\n', length(sig), length(sig)/fs);
        fprintf('  fs            : %.6g Hz   |   Nyquist: %.6g Hz\n', fs, nyquist);
        fprintf('  f1            : %.4g Hz  |   %d harmonic(s) requested\n', f1, num_harmonics);
        fprintf('  f2            : %.4g Hz\n', f2);
        fprintf('  Wavelet / level: %s  /  %d\n', options.Wavelet, options.WPTLevel);
        fprintf('  Chunk size    : %d samples   |   Padding: %d\n', options.ChunkSize, options.Padding);
        fprintf('  Output rows   : %s\n', strjoin(freq_labels, ',  '));
        fprintf('---\n');
    end

    % ------------------------------------------------------------------ %
    %  2.  Identify WPT sub-band indices using a representative segment
    % ------------------------------------------------------------------ %
    probe_len  = min(1e6, length(sig));
    probe_start = max(1, floor(length(sig)/2) - floor(probe_len/2));
    probe_seg  = sig(probe_start : probe_start + probe_len - 1);

    [~, ~, cfreq] = modwptdetails(probe_seg, options.Wavelet, options.WPTLevel);
    cfreq_scaled = cfreq * fs;                          % centre freqs in Hz
    bw           = mean(diff(cfreq)) * fs;              % approximate band width
    bb_lo        = cfreq_scaled - bw/2;
    bb_hi        = cfreq_scaled + bw/2;

    % indices for harmonics of f1
    ids_amp = zeros(1, num_harmonics);
    for k = 1:num_harmonics
        target = k * f1;
        row = find(target >= bb_lo & target <= bb_hi, 1);
        if isempty(row)
            [~, row] = min(abs(cfreq_scaled - target));
           warning('WT_AFM_harmonic_extraction:bandNotFound', ...
            ['Harmonic %d (%.4g Hz) does not fall exactly within any WPT sub-band. ' ...
            'Using nearest sub-band (centre %.4g Hz).'], ...
             k, target, cfreq_scaled(row));
        end
        ids_amp(k) = row;
    end

    % index for f2
    row_f2 = find(f2 >= bb_lo & f2 <= bb_hi, 1);
    if isempty(row_f2)
        [~, row_f2] = min(abs(cfreq_scaled - f2));
        warning('WT_AFM_harmonic_extraction:f2BandNotFound', ...
            ['f2 (%.4g Hz) does not fall exactly within any WPT sub-band. ' ...
            'Using nearest sub-band (centre %.4g Hz).'], ...
            f2, cfreq_scaled(row_f2));
    end
    ids_all = [ids_amp, row_f2];          % (num_harmonics+1) indices
    n_targets = length(ids_all);

    % ------------------------------------------------------------------ %
    %  3.  Pre-allocate outputs
    % ------------------------------------------------------------------ %
    N        = length(sig);
    amp_all  = nan(n_targets, N);
    phase_f1 = nan(1, N);

    pad    = options.Padding;
    chunk  = options.ChunkSize;
    bwExp  = options.BandwidthExp;
    bwHalf = (exp(bwExp) - 1) / 2;     % fractional half-bandwidth

    % ------------------------------------------------------------------ %
    %  4.  Main processing loop  (chunked to manage memory)
    % ------------------------------------------------------------------ %
    n_chunks   = ceil(N / chunk);
    a          = 0;                      % running sample offset

    for k = 1:n_chunks

        if options.Verbose
            fprintf('  Chunk %d / %d  ...', k, n_chunks);
        end

        % --- extract current chunk + padding ---
        is_last  = (a + chunk >= N);
        if ~is_last
            idx = a + (1:chunk);
        else
            idx = a+1 : N;
        end
        chunk_len = length(idx);

        q   = [zeros(1,pad),  sig(idx),  zeros(1,pad)];
        fdd = [zeros(1,pad),  fd(idx),   zeros(1,pad)];

        % --- WPT decomposition ---
        [wpt, ~, ~] = modwptdetails(q, options.Wavelet, options.WPTLevel);
        QQ = wpt(ids_all, :);           % (n_targets x padded_chunk)

        % --- CWT amplitude per target frequency ---
        for j = 1:n_targets

            if j <= num_harmonics
                f_target = j * f1;
            else
                f_target = f2;
            end

            % For f2 near Nyquist, use Nyquist as the upper limit
            if f_target > 0.9 * nyquist
                ff_lo = nyquist / 2^bwExp;
                ff_hi = nyquist;
            else
                ff_lo = f_target * (1 - bwHalf);
                ff_hi = f_target * (1 + bwHalf);
            end

            % Ensure limits are within (0, Nyquist)
            ff_lo = max(ff_lo, 1);
            ff_hi = min(ff_hi, nyquist * 0.9999);

            [cc, ff] = cwt(QQ(j,:), fs, 'FrequencyLimits', [ff_lo, ff_hi]);
            [~, id]  = min(abs(ff - f_target));

            amp_interim = abs(cc(id, :));
            amp_all(j, a+1 : a+chunk_len) = amp_interim(pad+1 : pad+chunk_len);
        end

        % --- Wavelet coherence phase at f1 ---
        ff1_lo = f1 * (1 - bwHalf);
        ff1_hi = f1 * (1 + bwHalf);
        [~, wcs1, ffdd1] = wcoherence(q, fdd, fs, 'FrequencyLimits', [ff1_lo, ff1_hi]);
        [~, id1] = min(abs(ffdd1 - f1));
        phase_f1(a+1 : a+chunk_len) = angle(wcs1(id1, pad+1 : pad+chunk_len));

        if options.Verbose
            fprintf(' done\n');
        end

        a = a + chunk_len;
    end

    if options.Verbose
        fprintf('=== Completed. Output: amp_all [%d x %d], phase_f1 [1 x %d] ===\n\n', ...
                size(amp_all,1), size(amp_all,2), length(phase_f1));
    end
end
